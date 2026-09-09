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

//! ORDER BY one column + LIMIT over hot rows, answered by walking the
//! multi-column index whose leading columns the WHERE pins, matches the
//! full sort: ASC and DESC, OFFSET, bounds on the order column, a predicate
//! the index does not cover, deleted and updated rows, rows written by the
//! same transaction, duplicate keys in a non-unique index, and hot rows
//! mixed with sealed ones.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;
use stoolap::common::I64Map;
use stoolap::core::{
    DataType, IndexEntry, IndexType, Operator, Result, Row, RowIdVec, Schema, Value,
};
use stoolap::storage::expression::{ComparisonExpr, Expression};
use stoolap::storage::index::MultiColumnIndex;
use stoolap::storage::traits::{Engine, Index};
use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

fn pairs(db: &Database, sql: &str) -> Vec<(i64, Option<i64>)> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<i64>(1).ok())
        })
        .collect()
}

/// The reference sorts every row by (t, id). The top-k must return the same
/// t sequence; when the t values in the answer are distinct the ids must
/// agree too (ties leave the order among equal keys to the path).
fn check(db: &Database, where_clause: &str, desc: bool, limit: usize, offset: usize) {
    let dir = if desc { "DESC" } else { "ASC" };
    let reference = pairs(
        db,
        &format!("SELECT id, t FROM c {where_clause} ORDER BY t {dir}, id {dir}"),
    );
    let expected: Vec<(i64, Option<i64>)> =
        reference.iter().skip(offset).take(limit).copied().collect();
    let offset_clause = if offset > 0 {
        format!(" OFFSET {offset}")
    } else {
        String::new()
    };
    let query =
        format!("SELECT id, t FROM c {where_clause} ORDER BY t {dir} LIMIT {limit}{offset_clause}");
    let actual = pairs(db, &query);
    let expected_t: Vec<Option<i64>> = expected.iter().map(|(_, t)| *t).collect();
    let actual_t: Vec<Option<i64>> = actual.iter().map(|(_, t)| *t).collect();
    assert_eq!(actual_t, expected_t, "{query}");
    let mut distinct = expected_t.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() == expected_t.len() {
        assert_eq!(actual, expected, "{query}");
    }
}

fn check_shapes(db: &Database) {
    for desc in [true, false] {
        check(db, "WHERE g = 'a' AND k = 'k3'", desc, 5, 0);
        check(db, "WHERE k = 'k3' AND g = 'a'", desc, 5, 3);
        check(db, "WHERE g = 'a' AND k = 'k3' AND t >= 300", desc, 4, 0);
        check(db, "WHERE g = 'a' AND k = 'k3' AND t > 300", desc, 4, 0);
        check(db, "WHERE g = 'a' AND k = 'k3' AND t <= 500", desc, 4, 0);
        check(db, "WHERE g = 'a' AND k = 'k3' AND t < 500", desc, 4, 2);
        check(
            db,
            "WHERE g = 'a' AND k = 'k3' AND t > 200 AND t < 700",
            desc,
            3,
            0,
        );
        // A predicate the index does not cover is checked on each row
        check(db, "WHERE g = 'a' AND k = 'k3' AND v > 50", desc, 6, 0);
        // Only the first index column pinned: the walk is over k, not t
        check(db, "WHERE g = 'b'", desc, 7, 0);
        // Past the end, and an empty group
        check(db, "WHERE g = 'a' AND k = 'k3'", desc, 1000, 0);
        check(db, "WHERE g = 'a' AND k = 'k99'", desc, 3, 0);
    }
}

fn fill(db: &Database) {
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY AUTO_INCREMENT, t INTEGER NOT NULL, \
         g TEXT NOT NULL, k TEXT NOT NULL, v INTEGER, UNIQUE(g, k, t))",
        (),
    )
    .unwrap();
    let mut stmt = String::from("INSERT INTO c (t, g, k, v) VALUES ");
    let mut first = true;
    for s in 0..100i64 {
        for k in 0..6 {
            for g in ["a", "b"] {
                if !first {
                    stmt.push(',');
                }
                first = false;
                // Out of time order, so the index order is not the insert order
                let t = (s * 37) % 1000 + 1;
                stmt.push_str(&format!("({t}, '{g}', 'k{k}', {})", (s * 7 + k) % 100));
            }
        }
    }
    db.execute(&stmt, ()).unwrap();
}

#[test]
fn test_hot_top_k_matches_the_full_sort() {
    let db = Database::open("memory://hot_top_k_shapes").unwrap();
    fill(&db);
    check_shapes(&db);

    // Deleted rows leave the walk; an update that moves a row to another
    // key leaves the old group and joins the new one
    db.execute("DELETE FROM c WHERE g = 'a' AND k = 'k3' AND t > 900", ())
        .unwrap();
    db.execute(
        "UPDATE c SET v = v + 1000 WHERE g = 'a' AND k = 'k3' AND t < 100",
        (),
    )
    .unwrap();
    db.execute(
        "UPDATE c SET k = 'k7' WHERE g = 'a' AND k = 'k3' AND t BETWEEN 100 AND 150",
        (),
    )
    .unwrap();
    check_shapes(&db);
}

#[test]
fn test_hot_top_k_inside_a_transaction_sees_its_own_rows() {
    let db = Database::open("memory://hot_top_k_txn").unwrap();
    fill(&db);
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "INSERT INTO c (t, g, k, v) VALUES (2, 'a', 'k3', 1), (995, 'a', 'k3', 2)",
        (),
    )
    .unwrap();
    db.execute("DELETE FROM c WHERE g = 'a' AND k = 'k3' AND t = 38", ())
        .unwrap();
    check_shapes(&db);
    db.execute("COMMIT", ()).unwrap();
    check_shapes(&db);
}

#[test]
fn test_hot_top_k_over_a_non_unique_index_keeps_the_key_order() {
    let db = Database::open("memory://hot_top_k_dups").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY AUTO_INCREMENT, t INTEGER NOT NULL, \
         k TEXT NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX c_k_t ON c (k, t)", ()).unwrap();
    let mut stmt = String::from("INSERT INTO c (t, k, v) VALUES ");
    for i in 0..600i64 {
        if i > 0 {
            stmt.push(',');
        }
        stmt.push_str(&format!("({}, 'k{}', {i})", (i * 13) % 50, i % 3));
    }
    db.execute(&stmt, ()).unwrap();
    // Ties on t: the sequence of t values must match the full sort
    for desc in [true, false] {
        let dir = if desc { "DESC" } else { "ASC" };
        let reference: Vec<i64> = ids(
            &db,
            &format!("SELECT t FROM c WHERE k = 'k1' ORDER BY t {dir}, id {dir}"),
        )
        .into_iter()
        .skip(4)
        .take(25)
        .collect();
        let actual = ids(
            &db,
            &format!("SELECT t FROM c WHERE k = 'k1' ORDER BY t {dir} LIMIT 25 OFFSET 4"),
        );
        assert_eq!(actual, reference, "{dir}");
        let distinct_rows = ids(
            &db,
            &format!("SELECT id FROM c WHERE k = 'k1' ORDER BY t {dir} LIMIT 25 OFFSET 4"),
        );
        let mut sorted = distinct_rows.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), distinct_rows.len(), "duplicate ids in {dir}");
    }
}

#[test]
fn test_hot_top_k_declines_a_nullable_order_column() {
    let db = Database::open("memory://hot_top_k_nullable").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY AUTO_INCREMENT, t INTEGER, g TEXT NOT NULL, \
         k TEXT NOT NULL, v INTEGER, UNIQUE(g, k, t))",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO c (t, g, k, v) VALUES (5, 'a', 'k3', 1), (NULL, 'a', 'k3', 2), \
         (1, 'a', 'k3', 3), (9, 'a', 'k3', 4), (NULL, 'a', 'k3', 5)",
        (),
    )
    .unwrap();
    for desc in [true, false] {
        check(&db, "WHERE g = 'a' AND k = 'k3'", desc, 3, 0);
        check(&db, "WHERE g = 'a' AND k = 'k3'", desc, 3, 2);
    }
}

#[test]
fn test_top_k_over_hot_and_sealed_rows_of_one_key() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/mixed", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    fill(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    // Newer and older hot rows for the same keys, a sealed row deleted, a
    // sealed row updated (its new version is hot, its sealed copy stale)
    db.execute(
        "INSERT INTO c (t, g, k, v) VALUES (1001, 'a', 'k3', 1), (1002, 'a', 'k3', 2), \
         (0, 'a', 'k3', 3), (1003, 'b', 'k0', 4)",
        (),
    )
    .unwrap();
    db.execute("DELETE FROM c WHERE g = 'a' AND k = 'k3' AND t = 38", ())
        .unwrap();
    db.execute(
        "UPDATE c SET v = -1 WHERE g = 'a' AND k = 'k3' AND t = 75",
        (),
    )
    .unwrap();
    check_shapes(&db);
    check(&db, "WHERE g = 'a' AND k = 'k3' AND v = -1", true, 3, 0);

    drop(db);
    let db = Database::open(&dsn).unwrap();
    check_shapes(&db);
    db.execute("INSERT INTO c (t, g, k, v) VALUES (1004, 'a', 'k3', 6)", ())
        .unwrap();
    check_shapes(&db);
}

#[test]
fn test_hot_top_k_with_bounds_that_hold_nothing_is_empty() {
    let db = Database::open("memory://hot_top_k_empty_bounds").unwrap();
    fill(&db);
    for where_clause in [
        "WHERE g = 'a' AND k = 'k3' AND t > 100 AND t < 50",
        "WHERE g = 'a' AND k = 'k3' AND t > 50 AND t < 50",
        "WHERE g = 'a' AND k = 'k3' AND t >= 50 AND t < 50",
        "WHERE g = 'a' AND k = 'k3' AND t > 50 AND t <= 50",
    ] {
        for desc in [true, false] {
            check(&db, where_clause, desc, 3, 0);
        }
    }
    check(
        &db,
        "WHERE g = 'a' AND k = 'k3' AND t >= 50 AND t <= 50",
        true,
        3,
        0,
    );
}

/// A snapshot sees the versions of its start; the index holds the keys of
/// the latest commits, so the walk does not answer for it
#[test]
fn test_snapshot_orders_by_the_values_it_sees() {
    let db = Database::open("memory://hot_top_k_snapshot").unwrap();
    fill(&db);
    let reader = db.clone();
    reader
        .execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())
        .unwrap();
    check(&reader, "WHERE g = 'a' AND k = 'k3'", true, 3, 0);
    // The newest row of the series moves to the front of time
    db.execute(
        "UPDATE c SET t = 0 WHERE g = 'a' AND k = 'k3' AND t = 1000",
        (),
    )
    .unwrap();
    check(&reader, "WHERE g = 'a' AND k = 'k3'", true, 3, 0);
    check(
        &reader,
        "WHERE g = 'a' AND k = 'k3' AND t > 900",
        false,
        3,
        0,
    );
    reader.execute("ROLLBACK", ()).unwrap();
    check(&db, "WHERE g = 'a' AND k = 'k3'", true, 3, 0);
}

/// Runs the first checkpoint from inside the WHERE. The legacy index path
/// inspects comparison metadata; a captured scan instead evaluates the frozen
/// hot rows. Both hooks exercise the same first-seal interleaving.
#[derive(Clone)]
struct SealInsideWhere {
    inner: ComparisonExpr,
    db: Database,
    fired: Arc<AtomicBool>,
}

impl SealInsideWhere {
    fn seal_once(&self) {
        if !self.fired.swap(true, Ordering::SeqCst) {
            self.db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
    }
}

impl std::fmt::Debug for SealInsideWhere {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SealInsideWhere")
    }
}

impl Expression for SealInsideWhere {
    fn evaluate(&self, row: &Row) -> Result<bool> {
        self.seal_once();
        self.inner.evaluate(row)
    }
    fn evaluate_fast(&self, row: &Row) -> bool {
        self.seal_once();
        self.inner.evaluate_fast(row)
    }
    fn with_aliases(&self, _: &rustc_hash::FxHashMap<String, String>) -> Box<dyn Expression> {
        self.clone_box()
    }
    fn prepare_for_schema(&mut self, schema: &Schema) {
        self.inner.prepare_for_schema(schema);
    }
    fn is_prepared(&self) -> bool {
        self.inner.is_prepared()
    }
    fn clone_box(&self) -> Box<dyn Expression> {
        Box::new(self.clone())
    }
    fn get_comparison_info(&self) -> Option<(&str, Operator, &Value)> {
        self.seal_once();
        self.inner.get_comparison_info()
    }
}

#[test]
fn test_first_seal_during_the_hot_top_k_keeps_the_series() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/seal", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, k TEXT NOT NULL, t INTEGER NOT NULL, UNIQUE(k, t))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, 'a', 100), (2, 'a', 50)", ())
        .unwrap();
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("c").unwrap();
    let fired = Arc::new(AtomicBool::new(false));
    let mut expr = SealInsideWhere {
        inner: ComparisonExpr::new("k", Operator::Eq, Value::text("a")),
        db: db.clone(),
        fired: Arc::clone(&fired),
    };
    expr.prepare_for_schema(table.schema());
    let answer = table.scan_top_k(Some(&expr), "t", false, 1, 0).unwrap();
    assert!(fired.load(Ordering::SeqCst));
    let got: Option<Vec<i64>> = answer.map(|rows| rows.into_iter().map(|(id, _)| id).collect());
    assert_eq!(got, Some(vec![1]));
    tx.rollback().unwrap();
}

/// Hands every call to the real index and, once, stops the caller right
/// after an add returned: the point in a commit where the index holds the
/// new key and the new version is not visible yet
struct StopAfterIndexAdd {
    inner: Arc<dyn Index>,
    armed: AtomicBool,
    stopped: mpsc::Sender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}

impl Index for StopAfterIndexAdd {
    fn memory_account(&self) -> Option<&stoolap::common::MemoryAccount> {
        self.inner.memory_account()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
    fn table_name(&self) -> &str {
        self.inner.table_name()
    }
    fn build(&mut self) -> Result<()> {
        Ok(())
    }
    fn add(&self, values: &[Value], row_id: i64, ref_id: i64) -> Result<()> {
        self.inner.add(values, row_id, ref_id)?;
        if self.armed.swap(false, Ordering::SeqCst) {
            self.stopped.send(()).unwrap();
            self.resume
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
        Ok(())
    }
    fn add_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        self.inner.add_batch(entries)
    }
    fn remove(&self, values: &[Value], row_id: i64, ref_id: i64) -> Result<()> {
        self.inner.remove(values, row_id, ref_id)
    }
    fn remove_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        self.inner.remove_batch(entries)
    }
    fn column_ids(&self) -> &[i32] {
        self.inner.column_ids()
    }
    fn column_names(&self) -> &[String] {
        self.inner.column_names()
    }
    fn data_types(&self) -> &[DataType] {
        self.inner.data_types()
    }
    fn index_type(&self) -> IndexType {
        self.inner.index_type()
    }
    fn is_unique(&self) -> bool {
        self.inner.is_unique()
    }
    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        self.inner.find(values)
    }
    fn find_range(
        &self,
        min: &[Value],
        max: &[Value],
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        self.inner
            .find_range(min, max, min_inclusive, max_inclusive)
    }
    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        self.inner.find_with_operator(op, values)
    }
    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> RowIdVec {
        self.inner.get_filtered_row_ids(expr)
    }
    fn walk_prefix_ordered(
        &self,
        prefix: &[Value],
        lower: Option<(&Value, bool)>,
        upper: Option<(&Value, bool)>,
        ascending: bool,
        visit: &mut dyn FnMut(i64, &Value) -> bool,
    ) -> bool {
        self.inner
            .walk_prefix_ordered(prefix, lower, upper, ascending, visit)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A commit moves the index key of the newest row before its new version
/// is visible; a reader in that window must answer from the visible values,
/// so the walk stands down while a commit publishes
#[test]
fn test_top_k_during_a_commit_answers_from_the_visible_versions() {
    let db = Database::open("memory://hot_top_k_commit_window").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, k TEXT NOT NULL, t INTEGER NOT NULL, UNIQUE(k, t))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, 'a', 100), (2, 'a', 50)", ())
        .unwrap();
    let store = db.engine().get_version_store("c").unwrap();
    let (index, _) = store.get_multi_column_index(&["k"]).unwrap();
    let name = index.name().to_string();
    let (stopped_tx, stopped_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    store
        .add_index(
            name,
            Arc::new(StopAfterIndexAdd {
                inner: index,
                armed: AtomicBool::new(true),
                stopped: stopped_tx,
                resume: Mutex::new(resume_rx),
            }),
        )
        .unwrap();
    let writer_db = db.clone();
    let writer = std::thread::spawn(move || {
        writer_db
            .execute("UPDATE c SET t = 0 WHERE id = 1", ())
            .unwrap();
    });
    stopped_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let full = ids(
        &db,
        "SELECT id FROM c WHERE k = 'a' ORDER BY t DESC, id DESC LIMIT 1",
    );
    let fast = ids(
        &db,
        "SELECT id FROM c WHERE k = 'a' ORDER BY t DESC LIMIT 1",
    );
    resume_tx.send(()).unwrap();
    writer.join().unwrap();
    assert_eq!(
        full,
        vec![1],
        "the old version stays visible until the commit completes"
    );
    assert_eq!(fast, full);
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM c WHERE k = 'a' ORDER BY t DESC LIMIT 1"
        ),
        vec![2]
    );
}

/// Two built groups are walked at the same time: a reader of one group
/// does not wait for a reader of another
#[test]
fn test_built_groups_are_walked_concurrently() {
    let index = Arc::new(MultiColumnIndex::new(
        "idx".into(),
        "c".into(),
        vec!["k".into(), "t".into()],
        vec![0, 1],
        vec![DataType::Integer; 2],
        false,
        0,
    ));
    for k in [1, 2] {
        index
            .add(&[Value::Integer(k), Value::Integer(100)], k, k)
            .unwrap();
        assert!(index.walk_prefix_ordered(&[Value::Integer(k)], None, None, true, &mut |_, _| true));
    }
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first_index = Arc::clone(&index);
    let first = std::thread::spawn(move || {
        first_index.walk_prefix_ordered(&[Value::Integer(1)], None, None, true, &mut |_, _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            false
        });
    });
    entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    let (second_tx, second_rx) = mpsc::channel();
    let second_index = Arc::clone(&index);
    let second = std::thread::spawn(move || {
        second_index.walk_prefix_ordered(&[Value::Integer(2)], None, None, true, &mut |_, _| {
            second_tx.send(()).unwrap();
            false
        });
    });
    let overlapped = second_rx.recv_timeout(Duration::from_secs(2)).is_ok();
    release_tx.send(()).unwrap();
    first.join().unwrap();
    second.join().unwrap();
    assert!(
        overlapped,
        "a built group's walk waited for another group's walk"
    );
}

/// A transaction that writes two tables is visible only once the whole
/// commit completes; the first table's index is updated long before that,
/// so its walk must stand down until the transaction is visible
#[test]
fn test_top_k_stands_down_until_a_two_table_commit_is_visible() {
    let db = Database::open("memory://hot_top_k_two_table_commit").unwrap();
    for table in ["c", "d"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k TEXT NOT NULL, t INTEGER NOT NULL, UNIQUE(k, t))"),
            (),
        )
        .unwrap();
        db.execute(
            &format!("INSERT INTO {table} VALUES (1, 'a', 100), (2, 'a', 50)"),
            (),
        )
        .unwrap();
    }
    let second_store = db.engine().get_version_store("d").unwrap();
    let (index, _) = second_store.get_multi_column_index(&["k"]).unwrap();
    let name = index.name().to_string();
    let (stopped_tx, stopped_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    second_store
        .add_index(
            name,
            Arc::new(StopAfterIndexAdd {
                inner: index,
                armed: AtomicBool::new(true),
                stopped: stopped_tx,
                resume: Mutex::new(resume_rx),
            }),
        )
        .unwrap();
    let writer_db = db.clone();
    let writer = std::thread::spawn(move || {
        writer_db.execute("BEGIN", ()).unwrap();
        writer_db
            .execute("UPDATE c SET t = 0 WHERE id = 1", ())
            .unwrap();
        writer_db
            .execute("UPDATE d SET t = 0 WHERE id = 1", ())
            .unwrap();
        writer_db.execute("COMMIT", ()).unwrap();
    });
    stopped_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    // c's table commit is done, d's is stopped, the transaction is not visible
    let full = ids(
        &db,
        "SELECT id FROM c WHERE k = 'a' ORDER BY t DESC, id DESC LIMIT 1",
    );
    let fast = ids(
        &db,
        "SELECT id FROM c WHERE k = 'a' ORDER BY t DESC LIMIT 1",
    );
    resume_tx.send(()).unwrap();
    writer.join().unwrap();
    assert_eq!(
        full,
        vec![1],
        "c's old row stays visible until the whole transaction commits"
    );
    assert_eq!(fast, full);
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM c WHERE k = 'a' ORDER BY t DESC LIMIT 1"
        ),
        vec![2]
    );
}
