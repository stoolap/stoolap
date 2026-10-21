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

//! A LIMIT subquery is prepared once per plan, read again on every
//! execution, and read in the snapshot the rest of its statement reads

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::prepared_dml_built;
use stoolap::Database;

/// The preparation counter is process-wide: every test here takes a turn
static TURNS: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    TURNS.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn a_prepared_limit_subquery_is_prepared_once() {
    let _turn = turn();
    let db = Database::open("memory://limit_subquery_prepare").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3), (4)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1), (2)", ()).unwrap();
    let statement = db
        .prepare("SELECT id FROM t ORDER BY id LIMIT (SELECT COUNT(*) FROM u)")
        .unwrap();
    let before = prepared_dml_built();
    let mut sizes = Vec::new();
    for id in 3..8 {
        sizes.push(statement.query(()).unwrap().count());
        db.execute("INSERT INTO u VALUES ($1)", (id,)).unwrap();
    }
    assert_eq!(sizes, [2, 3, 4, 4, 4], "each execution reads u again");
    assert_eq!(prepared_dml_built() - before, 1, "prepared again");
}

#[test]
fn a_limit_subquery_reads_the_snapshot_its_ctes_read() {
    let _turn = turn();
    let db = Database::open("memory://limit_subquery_snapshot").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
        .unwrap();
    db.set_default_isolation_level(stoolap::IsolationLevel::SnapshotIsolation)
        .unwrap();
    let writer = db.clone();
    stoolap::test_failpoints::before_row_counts_read(move || {
        writer.execute("DELETE FROM t WHERE id = 1", ()).unwrap();
    });
    let ids: Vec<i64> = db
        .query(
            "WITH c(id) AS (SELECT id FROM t) SELECT id FROM c ORDER BY id LIMIT (SELECT COUNT(*) FROM t)",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
            .unwrap(),
        2,
        "the hook did not run"
    );
    assert_eq!(
        ids,
        [1, 2, 3],
        "the CTE and the LIMIT read different states"
    );
}
