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

//! A table read through a statement view sees the commits complete when the
//! view was taken, in the hot store and in the volumes alike

#![cfg(feature = "test-failpoints")]

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use stoolap::core::{Operator, Value};
use stoolap::storage::expression::{ComparisonExpr, Expression};
use stoolap::storage::mvcc::registry::StatementView;
use stoolap::storage::traits::{Engine, Table, Transaction};
use stoolap::Database;

const WATCHDOG: Duration = Duration::from_secs(15);

#[derive(Clone, Copy)]
enum Stop {
    BeforeMarker,
    BeforeVisible,
}

/// Commits `sqls` in one transaction on another handle, stopped at `stop`
/// while `during` runs on this thread
fn while_committing(
    db: &Database,
    sqls: &'static [&'static str],
    stop: Stop,
    during: impl FnOnce(),
) {
    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let writer = db.clone();
    let commit = std::thread::spawn(move || {
        let mut tx = writer.begin().unwrap();
        for sql in sqls {
            tx.execute(sql, ()).unwrap();
        }
        let hook = move || {
            published_tx.send(()).unwrap();
            finish_rx.recv_timeout(WATCHDOG).unwrap();
        };
        match stop {
            Stop::BeforeMarker => stoolap::test_failpoints::before_commit_marker(hook),
            Stop::BeforeVisible => stoolap::test_failpoints::before_commit_visible(hook),
        }
        tx.commit().unwrap();
    });
    published_rx.recv_timeout(WATCHDOG).unwrap();
    during();
    finish_tx.send(()).unwrap();
    commit.join().unwrap();
}

/// A read-only transaction and its table of `t` read through `view`
fn through(db: &Database, view: &Arc<StatementView>) -> (Box<dyn Transaction>, Box<dyn Table>) {
    let txn = db.engine().begin_transaction().unwrap();
    let table = txn.get_table_with_view("t", Arc::clone(view)).unwrap();
    (txn, table)
}

/// (id, v) of the rows `table` reads, those with v = `v` when given
fn rows(table: &dyn Table, v: Option<i64>) -> Vec<(i64, i64)> {
    let filter = v.map(|v| {
        let mut filter = ComparisonExpr::new("v", Operator::Eq, Value::Integer(v));
        filter.prepare_for_schema(table.schema());
        filter
    });
    let mut rows: Vec<(i64, i64)> = table
        .collect_all_rows(filter.as_ref().map(|f| f as &dyn Expression))
        .unwrap()
        .into_iter()
        .map(|(id, row)| match row.get(1) {
            Some(Value::Integer(v)) => (id, *v),
            other => panic!("v of row {id}: {other:?}"),
        })
        .collect();
    rows.sort_unstable();
    rows
}

fn pair(id: i64, row: &stoolap::core::Row) -> (i64, i64) {
    match (row.get(0), row.get(1)) {
        (Some(Value::Integer(id)), Some(Value::Integer(v))) => (*id, *v),
        other => panic!("row {id}: {other:?}"),
    }
}

/// What `table` reads, and that every other read path and every aggregate
/// it answers agrees with it
fn state(table: &dyn Table) -> Vec<(i64, i64)> {
    let seen = rows(table, None);
    let mut scanned = Vec::new();
    let mut scanner = table.scan(&[0, 1], None).unwrap();
    while scanner.next() {
        scanned.push(pair(0, scanner.row()));
    }
    scanned.sort_unstable();
    assert_eq!(scanned, seen, "a scan");
    let mut limited: Vec<(i64, i64)> = table
        .collect_rows_with_limit(None, 100, 0)
        .unwrap()
        .into_iter()
        .map(|(id, row)| pair(id, &row))
        .collect();
    limited.sort_unstable();
    assert_eq!(limited, seen, "a limited read");
    let mut any = ComparisonExpr::new("id", Operator::Gte, Value::Integer(i64::MIN));
    any.prepare_for_schema(table.schema());
    let mut fetched: Vec<(i64, i64)> = table
        .fetch_rows_by_ids(&[1, 2, 3, 4, 5, 6, 7], &any)
        .unwrap()
        .into_iter()
        .map(|(id, row)| pair(id, &row))
        .collect();
    fetched.sort_unstable();
    assert_eq!(fetched, seen, "rows fetched by id");
    let mut looked_up = Vec::new();
    for id in 1..=7 {
        let mut key = ComparisonExpr::new("id", Operator::Eq, Value::Integer(id));
        key.prepare_for_schema(table.schema());
        let mut scanner = table.scan(&[0, 1], Some(&key)).unwrap();
        while scanner.next() {
            looked_up.push(pair(id, scanner.row()));
        }
    }
    assert_eq!(looked_up, seen, "rows looked up by primary key");
    assert_eq!(
        table.row_count().unwrap(),
        seen.len(),
        "row count of {seen:?}"
    );
    if let Some((sum, count)) = table.sum_column(1).unwrap() {
        let expected: i64 = seen.iter().map(|(_, v)| v).sum();
        assert_eq!(
            (sum as i64, count),
            (expected, seen.len()),
            "sum of {seen:?}"
        );
    }
    if let Some(min) = table.min_column(1).unwrap() {
        let expected = seen.iter().map(|(_, v)| *v).min().map(Value::Integer);
        assert_eq!(min, expected, "min of {seen:?}");
    }
    if let Some(max) = table.max_column(1).unwrap() {
        let expected = seen.iter().map(|(_, v)| *v).max().map(Value::Integer);
        assert_eq!(max, expected, "max of {seen:?}");
    }
    seen
}

fn open_file(dir: &Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap()
}

fn create(db: &Database) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
}

const CHANGES: &[&str] = &[
    "UPDATE t SET v = v + 100 WHERE id IN (1, 4)",
    "DELETE FROM t WHERE id IN (2, 5)",
    "INSERT INTO t VALUES (7, 70)",
];
const BEFORE: [(i64, i64); 6] = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];
const AFTER: [(i64, i64); 5] = [(1, 110), (3, 30), (4, 140), (6, 60), (7, 70)];

/// Rows 1..=3 sealed when `sealed`, rows 4..=6 hot
fn fill(db: &Database, sealed: bool) {
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    if sealed {
        db.engine().seal_for_test().unwrap();
    }
    db.execute("INSERT INTO t VALUES (4, 40), (5, 50), (6, 60)", ())
        .unwrap();
}

fn check_commit_window(db: &Database, shape: &str, stop: Stop) {
    let view = Rc::new(RefCell::new(None));
    let during = Rc::new(RefCell::new(Vec::new()));
    let (taken, seen) = (Rc::clone(&view), Rc::clone(&during));
    while_committing(db, CHANGES, stop, || {
        let captured = db.engine().registry().capture_view();
        let (_txn, table) = through(db, &captured);
        *seen.borrow_mut() = state(&*table);
        *taken.borrow_mut() = Some(captured);
    });
    let view = view.borrow_mut().take().unwrap();
    assert_eq!(
        *during.borrow(),
        BEFORE,
        "{shape}: read in the commit window"
    );
    let (_txn, table) = through(db, &view);
    assert_eq!(
        state(&*table),
        BEFORE,
        "{shape}: read after the commit completed"
    );
    drop((_txn, table, view));
    let fresh = db.engine().registry().capture_view();
    let (_txn, table) = through(db, &fresh);
    assert_eq!(
        state(&*table),
        AFTER,
        "{shape}: a view taken after the commit"
    );
}

#[test]
fn a_view_taken_in_a_hot_commit_window_keeps_the_state_before_the_commit() {
    for (stop, name) in [
        (Stop::BeforeMarker, "marker"),
        (Stop::BeforeVisible, "visible"),
    ] {
        let db = Database::open(&format!("memory://statement_view_reads_hot_{name}")).unwrap();
        create(&db);
        fill(&db, false);
        check_commit_window(&db, &format!("hot, before the {name}"), stop);
    }
}

#[test]
fn a_view_taken_in_a_commit_window_keeps_one_state_over_hot_and_sealed_rows() {
    for (stop, name) in [
        (Stop::BeforeMarker, "marker"),
        (Stop::BeforeVisible, "visible"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = open_file(dir.path());
        create(&db);
        fill(&db, true);
        check_commit_window(&db, &format!("hot and sealed, before the {name}"), stop);
    }
}

#[test]
fn a_view_keeps_the_rows_a_later_commit_deletes_and_changes() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open_file(dir.path());
        create(&db);
        fill(&db, sealed);
        let view = db.engine().registry().capture_view();
        let (_txn, table) = through(&db, &view);
        let mut tx = db.begin().unwrap();
        for sql in CHANGES {
            tx.execute(sql, ()).unwrap();
        }
        tx.commit().unwrap();
        assert_eq!(state(&*table), BEFORE, "sealed={sealed}");
    }
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn an_arena_window_taken_across_a_whole_commit_answers_from_the_chains() {
    let db = Database::open("memory://statement_view_reads_arena_epoch").unwrap();
    create(&db);
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    let view = db.engine().registry().capture_view();
    let (_txn, table) = through(&db, &view);
    let writer = db.clone();
    stoolap::test_failpoints::after_arena_epoch_read(move || {
        std::thread::spawn(move || {
            writer
                .execute("UPDATE t SET v = 99 WHERE id = 1", ())
                .unwrap();
        })
        .join()
        .unwrap();
    });
    let (sum, count) = table.sum_column(1).unwrap().unwrap();
    assert_eq!((sum, count), (60.0, 3), "the view excludes the update");
}

#[test]
fn a_snapshot_transaction_takes_no_statement_view() {
    let db = Database::open("memory://statement_view_reads_snapshot").unwrap();
    create(&db);
    let txn = db
        .engine()
        .begin_transaction_with_level(stoolap::core::IsolationLevel::SnapshotIsolation)
        .unwrap();
    let view = db.engine().registry().capture_view();
    assert!(txn.get_table_with_view("t", view).is_err());
}

#[test]
fn the_hot_version_a_view_sees_hides_the_sealed_copy_whatever_the_filter() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    create(&db);
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    db.engine().seal_for_test().unwrap();
    db.execute("UPDATE t SET v = 20 WHERE id = 1", ()).unwrap();
    let view = Rc::new(RefCell::new(None));
    let taken = Rc::clone(&view);
    while_committing(
        &db,
        &["UPDATE t SET v = 30 WHERE id = 1"],
        Stop::BeforeVisible,
        || {
            *taken.borrow_mut() = Some(db.engine().registry().capture_view());
        },
    );
    let view = view.borrow_mut().take().unwrap();
    let (_txn, table) = through(&db, &view);
    assert_eq!(state(&*table), vec![(1, 20)]);
    assert_eq!(rows(&*table, Some(10)), vec![], "the sealed copy came back");
    assert_eq!(rows(&*table, Some(20)), vec![(1, 20)]);
    assert_eq!(
        rows(&*table, Some(30)),
        vec![],
        "a commit the view excludes"
    );
    drop((_txn, table, view));
    let fresh = db.engine().registry().capture_view();
    let (_txn, table) = through(&db, &fresh);
    assert_eq!(state(&*table), vec![(1, 30)]);
}

/// SUM(v) of `t` read by `read` while a commit publishes between the arena
/// window opening and the pass over the heads
#[cfg(not(feature = "test-filedb"))]
fn sum_with_a_commit_inside_the_arena_pass(db: &Database, read: impl FnOnce() -> f64) -> f64 {
    let (published_tx, published_rx) = mpsc::channel();
    let (finish_tx, finish_rx) = mpsc::channel::<()>();
    let commit = Rc::new(RefCell::new(None));
    let (writer, handle) = (db.clone(), Rc::clone(&commit));
    stoolap::test_failpoints::after_arena_window_opened(move || {
        *handle.borrow_mut() = Some(std::thread::spawn(move || {
            let mut tx = writer.begin().unwrap();
            tx.execute("UPDATE t SET v = 99 WHERE id = 1", ()).unwrap();
            stoolap::test_failpoints::before_commit_visible(move || {
                published_tx.send(()).unwrap();
                finish_rx.recv_timeout(WATCHDOG).unwrap();
            });
            tx.commit().unwrap();
        }));
        published_rx.recv_timeout(WATCHDOG).unwrap();
    });
    let sum = read();
    finish_tx.send(()).unwrap();
    commit
        .borrow_mut()
        .take()
        .expect("the read opened an arena window")
        .join()
        .unwrap();
    sum
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn an_arena_pass_a_commit_published_into_answers_from_the_chains() {
    let db = Database::open("memory://statement_view_reads_arena_pass").unwrap();
    create(&db);
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    let sum = sum_with_a_commit_inside_the_arena_pass(&db, || {
        db.query_one::<f64, _>("SELECT SUM(v) FROM t", ()).unwrap()
    });
    assert_eq!(sum, 60.0, "without a view");
    db.execute("UPDATE t SET v = 10 WHERE id = 1", ()).unwrap();
    let view = db.engine().registry().capture_view();
    let (_txn, table) = through(&db, &view);
    let sum =
        sum_with_a_commit_inside_the_arena_pass(&db, || table.sum_column(1).unwrap().unwrap().0);
    assert_eq!(sum, 60.0, "through a view");
}

#[test]
fn a_hot_row_count_without_a_view_waits_out_a_publishing_delete() {
    let db = Database::open("memory://statement_view_reads_fast_count").unwrap();
    create(&db);
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    let txn = db.engine().begin_transaction().unwrap();
    let table = txn.get_table("t").unwrap();
    let mut counted = None;
    while_committing(
        &db,
        &["DELETE FROM t WHERE id = 1"],
        Stop::BeforeVisible,
        || {
            counted = Some(table.row_count().unwrap());
        },
    );
    assert_eq!(counted, Some(3), "the delete is not visible yet");
    assert_eq!(table.row_count().unwrap(), 2);
}

#[test]
fn a_seal_between_two_reads_of_a_view_changes_nothing_it_reads() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file(dir.path());
    create(&db);
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    let view = Rc::new(RefCell::new(None));
    let taken = Rc::clone(&view);
    while_committing(
        &db,
        &["UPDATE t SET v = 99 WHERE id = 1"],
        Stop::BeforeVisible,
        || {
            *taken.borrow_mut() = Some(db.engine().registry().capture_view());
        },
    );
    let view = view.borrow_mut().take().unwrap();
    let (_txn, table) = through(&db, &view);
    let first = state(&*table);
    assert_eq!(first, vec![(1, 10), (2, 20), (3, 30)]);
    db.engine().seal_for_test().unwrap();
    assert_eq!(state(&*table), first, "after a seal");
    drop((_txn, table, view));
    let fresh = db.engine().registry().capture_view();
    let (_txn, table) = through(&db, &fresh);
    let before = state(&*table);
    db.engine().seal_for_test().unwrap();
    assert_eq!(state(&*table), before, "a view with nothing in flight");
    assert_eq!(before, vec![(1, 99), (2, 20), (3, 30)]);
}
