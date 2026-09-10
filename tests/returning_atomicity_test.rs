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

use std::sync::{Mutex, MutexGuard};

use stoolap::api::Transaction;
use stoolap::core::{Result, Value};
use stoolap::functions::scalar::SleepFunction;
use stoolap::functions::{global_registry, FunctionInfo, ScalarFunction};
use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    db
}

fn values(db: &Database) -> Vec<(i64, i64)> {
    db.query("SELECT id, v FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect()
}

fn check_rollback(db: &Database, sql: &str) {
    let error = db.query(sql, ()).err().unwrap();
    assert!(
        error.to_string().contains("Invalid regular expression"),
        "{error}"
    );
    assert_eq!(values(db), [(1, 10), (2, 20)], "{sql}");
    let other = db.clone();
    assert_eq!(other.execute("UPDATE t SET v = v + 1", ()).unwrap(), 2);
    assert_eq!(values(db), [(1, 11), (2, 21)]);
}

#[test]
fn autocommit_insert_returning_error_rolls_back() {
    let db = setup("autocommit_insert_returning_error_rolls_back");
    check_rollback(&db, "INSERT INTO t VALUES (3, 30), (4, 40) RETURNING 'x' REGEXP CASE WHEN id = 4 THEN '[' ELSE 'x' END");
}

#[test]
fn autocommit_insert_select_returning_error_rolls_back() {
    let db = setup("autocommit_insert_select_returning_error_rolls_back");
    check_rollback(&db, "INSERT INTO t SELECT id + 2, v FROM t ORDER BY id RETURNING 'x' REGEXP CASE WHEN id = 4 THEN '[' ELSE 'x' END");
}

#[test]
fn autocommit_upsert_returning_error_rolls_back() {
    let db = setup("autocommit_upsert_returning_error_rolls_back");
    check_rollback(&db, "INSERT INTO t VALUES (1, 30), (2, 40) ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v RETURNING 'x' REGEXP CASE WHEN id = 2 THEN '[' ELSE 'x' END");
}

#[test]
fn autocommit_upsert_select_returning_error_rolls_back() {
    let db = setup("autocommit_upsert_select_returning_error_rolls_back");
    check_rollback(&db, "INSERT INTO t SELECT id, v + 20 FROM t ORDER BY id ON CONFLICT (id) DO UPDATE SET v = EXCLUDED.v RETURNING 'x' REGEXP CASE WHEN id = 2 THEN '[' ELSE 'x' END");
}

#[test]
fn autocommit_update_returning_error_rolls_back() {
    let db = setup("autocommit_update_returning_error_rolls_back");
    check_rollback(
        &db,
        "UPDATE t SET v = v + 20 RETURNING 'x' REGEXP CASE WHEN id = 2 THEN '[' ELSE 'x' END",
    );
}

#[test]
fn autocommit_delete_returning_error_rolls_back() {
    let db = setup("autocommit_delete_returning_error_rolls_back");
    check_rollback(
        &db,
        "DELETE FROM t RETURNING 'x' REGEXP CASE WHEN id = 2 THEN '[' ELSE 'x' END",
    );
}

#[test]
fn prepared_returning_error_leaves_autocommit_reusable() {
    let db = setup("prepared_returning_error_leaves_autocommit_reusable");
    let insert = db
        .prepare("INSERT INTO t VALUES ($1, $2) RETURNING id, v, 'x' REGEXP $3")
        .unwrap();
    for id in [3, 4] {
        let error = insert.query((id, id * 10, "[")).err().unwrap();
        assert!(
            error.to_string().contains("Invalid regular expression"),
            "{error}"
        );
        assert_eq!(values(&db), [(1, 10), (2, 20)]);
    }
    for id in [3, 4] {
        let mut rows = insert.query((id, id * 10, "x")).unwrap();
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get::<i64>(0).unwrap(), id);
        assert_eq!(row.get::<i64>(1).unwrap(), id * 10);
        assert!(row.get::<bool>(2).unwrap());
        assert!(rows.next().is_none());
    }
    assert_eq!(values(&db), [(1, 10), (2, 20), (3, 30), (4, 40)]);
}

#[test]
fn failed_cold_delete_returning_restores_cascade_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/returning", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES t(id) ON DELETE CASCADE)", ()).unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        db.execute("INSERT INTO children VALUES (1, 1), (2, 2)", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        check_rollback(
            &db,
            "DELETE FROM t RETURNING 'x' REGEXP CASE WHEN id = 2 THEN '[' ELSE 'x' END",
        );
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM children", ())
                .unwrap(),
            2
        );
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(values(&db), [(1, 11), (2, 21)]);
    assert_eq!(
        db.query_one::<i64, _>("SELECT SUM(parent_id) FROM children", ())
            .unwrap(),
        3
    );
}

static HOOK_LOCK: Mutex<()> = Mutex::new(());
static HOOK_STATE: Mutex<(Option<Transaction>, usize)> = Mutex::new((None, 0));

#[derive(Default)]
struct CommitDuringReturning;

impl ScalarFunction for CommitDuringReturning {
    fn name(&self) -> &str {
        "SLEEP"
    }

    fn info(&self) -> FunctionInfo {
        SleepFunction.info()
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        let transaction = {
            let mut state = HOOK_STATE.lock().unwrap();
            state.1 += 1;
            state.0.take()
        };
        if let Some(mut transaction) = transaction {
            transaction.commit().unwrap();
        }
        Ok(args[0].clone())
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Self)
    }
}

struct ReturningHook {
    _serial: MutexGuard<'static, ()>,
}

impl ReturningHook {
    fn new(transaction: Transaction) -> Self {
        let serial = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        *HOOK_STATE.lock().unwrap() = (Some(transaction), 0);
        global_registry().register_scalar::<CommitDuringReturning>();
        Self { _serial: serial }
    }

    fn calls(&self) -> usize {
        HOOK_STATE.lock().unwrap().1
    }
}

impl Drop for ReturningHook {
    fn drop(&mut self) {
        global_registry().register_scalar::<SleepFunction>();
        let transaction = HOOK_STATE.lock().unwrap().0.take();
        drop(transaction);
    }
}

fn check_commit_conflict(name: &str, source: &str, upsert: bool) {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, u INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    let mut competing = db.begin().unwrap();
    competing
        .execute("INSERT INTO t VALUES (1, 7, 50)", ())
        .unwrap();
    let hook = ReturningHook::new(competing);
    let action = if upsert {
        "DO UPDATE SET v = 20"
    } else {
        "DO NOTHING"
    };
    let mut rows = db
        .query(
            &format!("INSERT INTO t {source} ON CONFLICT (u) {action} RETURNING id, v, SLEEP(v)"),
            (),
        )
        .unwrap();
    assert_eq!(rows.columns(), ["id", "v", "SLEEP(v)"]);
    if upsert {
        let row = rows.next().unwrap().unwrap();
        assert_eq!(row.get::<i64>(0).unwrap(), 1);
        assert_eq!(row.get::<i64>(1).unwrap(), 20);
        assert_eq!(row.get::<i64>(2).unwrap(), 20);
        assert_eq!(hook.calls(), 2);
        assert_eq!(values(&db), [(1, 20)]);
    } else {
        assert_eq!(hook.calls(), 1);
        assert_eq!(values(&db), [(1, 50)]);
    }
    assert!(rows.next().is_none());
}

#[test]
fn do_nothing_discards_prepared_returning_after_commit_conflict() {
    check_commit_conflict(
        "do_nothing_discards_prepared_returning_after_commit_conflict",
        "VALUES (2, 7, 10)",
        false,
    );
}

#[test]
fn select_do_nothing_discards_prepared_returning_after_commit_conflict() {
    check_commit_conflict(
        "select_do_nothing_discards_prepared_returning_after_commit_conflict",
        "SELECT 2, 7, 10",
        false,
    );
}

#[test]
fn upsert_retry_replaces_prepared_returning_after_commit_conflict() {
    check_commit_conflict(
        "upsert_retry_replaces_prepared_returning_after_commit_conflict",
        "VALUES (2, 7, 10)",
        true,
    );
}

#[test]
fn select_upsert_retry_replaces_prepared_returning_after_commit_conflict() {
    check_commit_conflict(
        "select_upsert_retry_replaces_prepared_returning_after_commit_conflict",
        "SELECT 2, 7, 10",
        true,
    );
}
