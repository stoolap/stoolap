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

//! A commit whose WAL marker fails leaves the database as it was: a sealed
//! row the failed statement updated or deleted stays visible, the indexes
//! that keep sealed rows keep its old keys, and a later statement finds the
//! table as before. The hot shapes are the control. A failed WAL write
//! poisons the WAL until the database is reopened, so the statements that
//! follow a failure run after a reopen.

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering;

use stoolap::storage::index::hnsw::HnswIndex;
use stoolap::test_failpoints;
use stoolap::Database;

fn dsn(dir: &tempfile::TempDir) -> String {
    format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    )
}

fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap())
        })
        .collect()
}

/// Runs `statement` with the WAL fsync failing; it must fail
fn failing(db: &Database, statement: &str) {
    test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
    let result = db.execute(statement, ());
    test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
    assert!(result.is_err(), "{statement} must fail with the marker");
}

/// A sealed row updated or deleted by a commit whose marker fails stays
/// visible, through a scan and through its key, now and after reopen
#[test]
fn a_failed_commit_leaves_the_sealed_row_visible() {
    let _guard = test_failpoints::FailpointGuard::new();
    for sealed in [false, true] {
        for statement in [
            "UPDATE t SET v = 99 WHERE id = 1",
            "DELETE FROM t WHERE id = 1",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let db = Database::open(&dsn(&dir)).unwrap();
            db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
                .unwrap();
            db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
                .unwrap();
            if sealed {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            let before = vec![(1, 10), (2, 20)];
            assert_eq!(pairs(&db, "SELECT id, v FROM t ORDER BY id"), before);

            failing(&db, statement);
            assert_eq!(
                pairs(&db, "SELECT id, v FROM t ORDER BY id"),
                before,
                "sealed={sealed} {statement}: the scan"
            );
            assert_eq!(
                pairs(&db, "SELECT id, v FROM t WHERE id = 1"),
                vec![(1, 10)],
                "sealed={sealed} {statement}: the key"
            );
            assert_eq!(
                pairs(&db, "SELECT COUNT(*), SUM(v) FROM t"),
                vec![(2, 30)],
                "sealed={sealed} {statement}: the count and the sum"
            );
            db.close().unwrap();
            drop(db);

            let db = Database::open(&dsn(&dir)).unwrap();
            assert_eq!(
                pairs(&db, "SELECT id, v FROM t ORDER BY id"),
                before,
                "sealed={sealed} {statement}: after reopen"
            );
            // The table takes the next statement as if nothing had happened
            db.execute("UPDATE t SET v = 11 WHERE id = 1", ()).unwrap();
            assert_eq!(
                pairs(&db, "SELECT id, v FROM t ORDER BY id"),
                vec![(1, 11), (2, 20)]
            );
            db.close().unwrap();
        }
    }
}

/// A row updated past the chain's history limit: the version the failed
/// commit displaced is restored although the chain no longer held it
#[test]
fn a_failed_commit_restores_a_row_past_its_history_limit() {
    let _guard = test_failpoints::FailpointGuard::new();
    for statement in [
        "UPDATE t SET v = 99 WHERE id = 1",
        "DELETE FROM t WHERE id = 1",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(&dir)).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 0), (2, 20)", ())
            .unwrap();
        for v in 1..=12 {
            db.execute(&format!("UPDATE t SET v = {v} WHERE id = 1"), ())
                .unwrap();
        }
        failing(&db, statement);
        assert_eq!(
            pairs(&db, "SELECT id, v FROM t ORDER BY id"),
            vec![(1, 12), (2, 20)],
            "{statement}: the last committed update stays"
        );
        assert_eq!(pairs(&db, "SELECT COUNT(*), SUM(v) FROM t"), vec![(2, 32)]);
        db.close().unwrap();
    }
}

/// A unique key the failed commit would have moved stays where it was:
/// the old value is still taken, the new one is free
#[test]
fn a_failed_commit_leaves_the_unique_index_as_it_was() {
    let _guard = test_failpoints::FailpointGuard::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(&dir)).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("CREATE UNIQUE INDEX idx_t_v ON t(v)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        failing(&db, "UPDATE t SET v = 99 WHERE id = 1");
        assert_eq!(
            pairs(&db, "SELECT id, v FROM t WHERE v = 10"),
            vec![(1, 10)],
            "sealed={sealed}: the old key still finds the row"
        );
        assert_eq!(pairs(&db, "SELECT id, v FROM t WHERE v = 99"), vec![]);
        db.close().unwrap();
        drop(db);
        let db = Database::open(&dsn(&dir)).unwrap();
        assert!(
            db.execute("INSERT INTO t VALUES (3, 10)", ()).is_err(),
            "sealed={sealed}: 10 is still row 1's"
        );
        db.execute("INSERT INTO t VALUES (4, 99)", ()).unwrap();
        assert_eq!(
            pairs(&db, "SELECT id, v FROM t ORDER BY id"),
            vec![(1, 10), (2, 20), (4, 99)]
        );
        db.close().unwrap();
    }
}

/// The HNSW graph keeps sealed rows, so a failed replacement of a sealed
/// vector, by another vector or by NULL, and a failed delete leave the old
/// vector searchable; a hot row is the control
#[test]
fn a_failed_commit_leaves_the_sealed_vector_in_the_graph() {
    let _guard = test_failpoints::FailpointGuard::new();
    for sealed in [false, true] {
        for statement in [
            "UPDATE t SET v = '[100,0]' WHERE id = 1",
            "UPDATE t SET v = NULL WHERE id = 1",
            "UPDATE t SET v = '[100,0]' WHERE k = 1",
            "DELETE FROM t WHERE id = 1",
            "DELETE FROM t WHERE id >= 1",
            "DELETE FROM t WHERE k = 1",
            "DELETE FROM t",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let db = Database::open(&dsn(&dir)).unwrap();
            db.execute(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v VECTOR(2))",
                (),
            )
            .unwrap();
            db.execute("INSERT INTO t VALUES (1, 1, '[1,0]'), (2, 2, '[10,0]')", ())
                .unwrap();
            db.execute("CREATE INDEX idx_t_v ON t(v) USING HNSW", ())
                .unwrap();
            if sealed {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            let nearest = "SELECT id FROM t ORDER BY VEC_DISTANCE_L2(v, '[1,0]') LIMIT 2";
            assert_eq!(nearest_ids(&db, nearest), vec![1, 2]);
            assert_eq!(graph_ids(&db), vec![1, 2]);

            failing(&db, statement);
            assert_eq!(
                graph_ids(&db),
                vec![1, 2],
                "sealed={sealed} {statement}: the graph itself"
            );
            assert_eq!(
                nearest_ids(&db, nearest),
                vec![1, 2],
                "sealed={sealed} {statement}: the query"
            );
            db.close().unwrap();
            drop(db);
            // The row is still there to be replaced for real
            let db = Database::open(&dsn(&dir)).unwrap();
            assert_eq!(nearest_ids(&db, nearest), vec![1, 2]);
            db.execute("UPDATE t SET v = '[100,0]' WHERE id = 1", ())
                .unwrap();
            assert_eq!(nearest_ids(&db, nearest), vec![2, 1]);
            db.close().unwrap();
        }
    }
}

fn nearest_ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// The ids the HNSW graph itself answers for `[1,0]`, nearest first
fn graph_ids(db: &Database) -> Vec<i64> {
    let store = db.engine().get_version_store("t").unwrap();
    let index = store.get_index("idx_t_v").unwrap();
    let graph = index.as_any().downcast_ref::<HnswIndex>().unwrap();
    let query: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    graph
        .search_nearest(&query, 2, 100)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// A vector index created while a transaction holds a sealed row's
/// replacement is refused, since that transaction kept nothing for the
/// index to take back; once the transaction is done the index is created
/// and the graph is right
#[test]
fn a_vector_index_is_not_created_over_a_pending_sealed_write() {
    let _guard = test_failpoints::FailpointGuard::new();
    for statement in [
        "UPDATE t SET v = '[100,0]' WHERE id = 1",
        "DELETE FROM t WHERE id = 1",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dsn(&dir)).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(2))", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, '[1,0]'), (2, '[10,0]')", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        let mut writer = db.begin().unwrap();
        writer.execute(statement, ()).unwrap();
        let created = db.execute("CREATE INDEX idx_t_v ON t(v) USING HNSW", ());
        assert!(
            matches!(
                created,
                Err(stoolap::core::Error::TableHasActiveTransactions)
            ),
            "{statement}: {created:?}"
        );
        writer.commit().unwrap();
        db.execute("CREATE INDEX idx_t_v ON t(v) USING HNSW", ())
            .unwrap();
        let expected = if statement.starts_with("UPDATE") {
            vec![2, 1]
        } else {
            vec![2]
        };
        assert_eq!(
            graph_ids(&db),
            expected,
            "{statement}: the graph after the write"
        );
        db.close().unwrap();
    }
}

/// A commit whose second table fails to apply leaves nothing of the first
/// table visible: the rows the transaction changed are as they were, the
/// row another transaction committed meanwhile stays, now and after reopen
#[test]
fn a_commit_that_fails_on_a_later_table_leaves_the_earlier_ones_as_they_were() {
    let _guard = test_failpoints::FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(&dir)).unwrap();
    for table in ["a", "b"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, v INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!("CREATE UNIQUE INDEX idx_{table}_v ON {table}(v)"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO {table} VALUES (1, 10), (2, 20)"), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // When the first table has published its indexes, another transaction
    // takes the key the second table is about to publish
    let other = db.clone();
    test_failpoints::after_indexes_published(move || {
        for table in ["a", "b"] {
            let _ = other.execute(&format!("INSERT INTO {table} VALUES (3, 999)"), ());
        }
    });
    let mut writer = db.begin().unwrap();
    writer
        .execute("UPDATE a SET v = 999 WHERE id = 1", ())
        .unwrap();
    writer
        .execute("UPDATE b SET v = 999 WHERE id = 1", ())
        .unwrap();
    assert!(writer.commit().is_err(), "the second table's key is taken");

    let check = |db: &Database, when: &str| {
        for table in ["a", "b"] {
            let rows = pairs(db, &format!("SELECT id, v FROM {table} ORDER BY id"));
            assert_eq!(
                rows[..2],
                [(1, 10), (2, 20)],
                "{when}: {table} is as it was"
            );
            assert!(rows.len() <= 3 && rows.get(2).is_none_or(|row| *row == (3, 999)));
            let taken = rows.len() == 3;
            let sum = if taken { 1029 } else { 30 };
            assert_eq!(
                pairs(db, &format!("SELECT COUNT(*), SUM(v) FROM {table}")),
                vec![(rows.len() as i64, sum)],
                "{when}: {table} counts as it was"
            );
            // The key the failed commit would have taken is free unless the
            // other transaction took it
            let free = db.execute(&format!("INSERT INTO {table} VALUES (4, 999)"), ());
            assert_eq!(free.is_ok(), !taken, "{when}: {table} key 999: {free:?}");
            if !taken {
                db.execute(&format!("DELETE FROM {table} WHERE id = 4"), ())
                    .unwrap();
            }
        }
    };
    check(&db, "before reopen");
    db.close().unwrap();
    drop(db);
    let db = Database::open(&dsn(&dir)).unwrap();
    check(&db, "after reopen");
}
