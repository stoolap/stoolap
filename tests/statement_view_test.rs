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

//! Every read of one statement sees the same committed state

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::Database;

const QUERY: &str =
    "SELECT id FROM t WHERE b = 1 AND a > (SELECT v FROM u WHERE u.id = t.id) ORDER BY id";

fn setup(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 1), (2, 20, 1), (5, 50, 1)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 5), (2, 25), (5, 60)", ())
        .unwrap();
    db
}

/// Runs `QUERY` with a commit by another handle landing while the first
/// outer row's subquery opens `u`, and returns the ids it gave
fn ids_with_commit_inside(db: &Database) -> Vec<i64> {
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::after_select_table_opened(move || {
        stoolap::test_failpoints::after_select_table_opened(move || {
            other
                .execute("UPDATE u SET v = 0 WHERE id IN (2, 5)", ())
                .unwrap();
            flag.set(true);
        });
    });
    let ids = db
        .query(QUERY, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert!(
        committed.get(),
        "the commit did not land inside the statement"
    );
    ids
}

#[test]
fn test_read_committed_statement_keeps_one_view() {
    let db = setup("memory://statement_view_rc");
    assert_eq!(ids_with_commit_inside(&db), vec![1]);
}

#[test]
fn test_snapshot_statement_keeps_one_view() {
    let db = setup("memory://statement_view_snapshot");
    db.execute("SET isolation_level = 'SNAPSHOT'", ()).unwrap();
    assert_eq!(ids_with_commit_inside(&db), vec![1]);
}

#[test]
fn test_read_committed_statement_in_transaction_keeps_one_view() {
    let db = setup("memory://statement_view_rc_tx");
    db.execute("BEGIN", ()).unwrap();
    assert_eq!(ids_with_commit_inside(&db), vec![1]);
    db.execute("COMMIT", ()).unwrap();
}

#[test]
fn test_read_committed_next_statement_sees_the_commit() {
    let db = setup("memory://statement_view_rc_next");
    assert_eq!(ids_with_commit_inside(&db), vec![1]);
    let after: Vec<i64> = db
        .query(QUERY, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(after, vec![1, 2, 5]);
}

fn remaining_ids(db: &Database) -> Vec<i64> {
    db.query("SELECT id FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// Runs `sql` with another handle's commit landing while the statement's
/// first scan has opened its table
fn run_with_commit_inside(db: &Database, sql: &str, commit: &'static str) {
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::after_select_table_opened(move || {
        other.execute(commit, ()).unwrap();
        flag.set(true);
    });
    db.execute(sql, ()).unwrap();
    assert!(
        committed.get(),
        "the commit did not land inside the statement"
    );
}

#[test]
fn test_read_committed_delete_subqueries_keep_one_view() {
    let db = setup("memory://statement_view_rc_delete");
    let committed = Rc::new(Cell::new(false));
    let (other, flag) = (db.clone(), Rc::clone(&committed));
    stoolap::test_failpoints::after_select_table_opened(move || {
        stoolap::test_failpoints::after_select_table_opened(move || {
            other
                .execute("UPDATE u SET v = 0 WHERE id IN (2, 5)", ())
                .unwrap();
            flag.set(true);
        });
    });
    db.execute(
        "DELETE FROM t WHERE b = 1 AND a > (SELECT v FROM u WHERE u.id = t.id)",
        (),
    )
    .unwrap();
    assert!(
        committed.get(),
        "the commit did not land inside the statement"
    );
    assert_eq!(remaining_ids(&db), vec![2, 5]);
}

const UPDATE_TARGET: &str = "UPDATE t SET b = 2 WHERE a IN (SELECT v FROM u)";

fn updated_ids(db: &Database) -> Vec<i64> {
    db.query("SELECT id FROM t WHERE b = 2 ORDER BY id", ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn test_read_committed_update_target_reads_the_newest_state() {
    let db = setup("memory://statement_view_rc_update_target");
    db.execute("INSERT INTO u VALUES (7, 70)", ()).unwrap();
    run_with_commit_inside(&db, UPDATE_TARGET, "INSERT INTO t VALUES (7, 70, 1)");
    assert_eq!(updated_ids(&db), vec![7]);
}

#[test]
fn test_read_committed_update_target_in_transaction_reads_the_newest_state() {
    let db = setup("memory://statement_view_rc_update_target_tx");
    db.execute("INSERT INTO u VALUES (7, 70)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    run_with_commit_inside(&db, UPDATE_TARGET, "INSERT INTO t VALUES (7, 70, 1)");
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(updated_ids(&db), vec![7]);
}

#[test]
fn a_statement_snapshot_the_caller_gave_reads_at_its_own_point() {
    use stoolap::core::IsolationLevel;
    use stoolap::executor::context::StatementSnapshot;
    use stoolap::storage::traits::Engine;
    let db = setup("memory://statement_view_given_snapshot");
    db.execute("INSERT INTO u VALUES (8, 1)", ()).unwrap();
    let engine = db.engine();
    let snapshot = StatementSnapshot::new(
        engine
            .begin_transaction_with_level(IsolationLevel::SnapshotIsolation)
            .unwrap(),
        IsolationLevel::SnapshotIsolation,
        None,
    );
    let viewed = StatementSnapshot::new(
        engine.begin_transaction().unwrap(),
        IsolationLevel::ReadCommitted,
        Some(engine.registry().capture_view()),
    );
    // Row 8 shows a later outer scan, row 2 a later subquery
    db.execute("INSERT INTO t VALUES (8, 80, 1)", ()).unwrap();
    db.execute("UPDATE u SET v = 0 WHERE id = 2", ()).unwrap();
    let executor = stoolap::Executor::new(std::sync::Arc::clone(engine));
    let mut wrong = Vec::new();
    for (name, snapshot) in [("snapshot", snapshot), ("view", viewed)] {
        let ctx = stoolap::ExecutionContext::new().with_statement_snapshot(snapshot);
        let seen: stoolap::Result<Vec<i64>> =
            executor
                .execute_with_context(QUERY, &ctx)
                .map(|mut result| {
                    let mut ids = Vec::new();
                    while result.next() {
                        ids.push(result.row().get(0).and_then(|v| v.as_int64()).unwrap());
                    }
                    ids
                });
        if seen.as_deref() != Ok(&[1]) {
            wrong.push(format!("{name}: {seen:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
