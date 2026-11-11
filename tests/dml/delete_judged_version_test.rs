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

//! A DELETE that finds its rows before it deletes them asks its WHERE of
//! the version it deletes, when another transaction commits in between

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex, MutexGuard};

use stoolap::functions::{
    global_registry, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
    ScalarFunction,
};
use stoolap::test_failpoints::{after_commit_counted, after_delete_rows_scanned, under_seal_fence};
use stoolap::{Database, Value};

/// The tests that count the judge's calls run one at a time
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `t` of three rows, two of them with `v = 10`, on the hot store and then sealed
fn each_layout(name: &str, extra: &[&str], check: impl Fn(&Database)) {
    let open = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
            (),
        )
        .unwrap();
        db.execute(
            "INSERT INTO t VALUES (1, 10, 'a'), (2, 10, 'b'), (3, 30, 'c')",
            (),
        )
        .unwrap();
        for sql in extra {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&open(&format!("memory://delete_judged_{name}"), false));
    let dir = tempfile::tempdir().unwrap();
    check(&open(&format!("file://{}", dir.path().display()), true));
}

fn rows(db: &Database) -> Vec<(i64, i64, String)> {
    db.query("SELECT id, v, s FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get(0).unwrap(),
                row.get(1).unwrap(),
                row.get(2).unwrap(),
            )
        })
        .collect()
}

/// Has another handle run `sql` once the next DELETE found its rows; the
/// cell says whether it ran
fn commit_between(db: &Database, sql: &'static str) -> Rc<Cell<bool>> {
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook = Rc::clone(&fired);
    after_delete_rows_scanned(move || {
        other.execute(sql, ()).unwrap();
        hook.set(true);
    });
    fired
}

/// Each WHERE the storage layer cannot take whole, as text and with a parameter
const NOT_PUSHED: [&str; 2] = [
    "DELETE FROM t WHERE v + 0 = 10",
    "DELETE FROM t WHERE v + 0 = $1",
];

fn delete(db: &Database, sql: &str) -> stoolap::Result<i64> {
    match sql.contains("$1") {
        true => db.execute(sql, (10,)),
        false => db.execute(sql, ()),
    }
}

#[test]
fn a_row_changed_to_fail_the_where_since_the_scan_is_not_deleted() {
    for sql in NOT_PUSHED {
        each_layout("fails", &[], |db| {
            let fired = commit_between(db, "UPDATE t SET v = 20 WHERE id = 1");
            assert_eq!(delete(db, sql).unwrap(), 1, "{sql}");
            assert!(fired.get(), "{sql}: the commit did not come between");
            assert_eq!(
                rows(db),
                [(1, 20, "a".into()), (3, 30, "c".into())],
                "{sql}"
            );
        });
    }
}

#[test]
fn a_row_changed_since_the_scan_that_still_passes_is_deleted_and_returned_as_deleted() {
    each_layout("passes", &[], |db| {
        let fired = commit_between(db, "UPDATE t SET s = 'z' WHERE id = 1");
        let returned: Vec<(i64, String)> = db
            .query("DELETE FROM t WHERE v + 0 = 10 RETURNING id, s", ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (row.get(0).unwrap(), row.get(1).unwrap())
            })
            .collect();
        assert!(fired.get(), "the commit did not come between");
        assert_eq!(returned, [(1, "z".into()), (2, "b".into())]);
        assert_eq!(rows(db), [(3, 30, "c".into())]);
    });
}

#[test]
fn a_delete_a_storage_filter_takes_whole_but_returns_rows_asks_the_deleted_version() {
    each_layout("returning", &[], |db| {
        let fired = commit_between(db, "UPDATE t SET v = 20 WHERE id = 1");
        let returned: Vec<i64> = db
            .query("DELETE FROM t WHERE v = 10 RETURNING id", ())
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert!(fired.get(), "the commit did not come between");
        assert_eq!(returned, [2]);
        assert_eq!(rows(db), [(1, 20, "a".into()), (3, 30, "c".into())]);
    });
}

#[test]
fn a_parent_row_changed_since_the_scan_refuses_the_delete() {
    let child = [
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES t(id) ON DELETE CASCADE)",
        "INSERT INTO c VALUES (1, 1), (2, 2)",
    ];
    for sql in NOT_PUSHED {
        each_layout("parent", &child, |db| {
            let fired = commit_between(db, "UPDATE t SET v = 20 WHERE id = 1");
            let error = delete(db, sql).unwrap_err().to_string();
            assert!(fired.get(), "{sql}: the commit did not come between");
            assert!(error.contains("write conflict"), "{sql}: {error}");
            assert_eq!(rows(db).len(), 3, "{sql}");
            assert_eq!(
                db.query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
                    .unwrap(),
                2,
                "{sql}"
            );
        });
    }
}

#[test]
fn a_table_without_a_primary_key_asks_the_deleted_version() {
    let dir = tempfile::tempdir().unwrap();
    for (dsn, sealed) in [
        ("memory://delete_judged_no_key".to_string(), false),
        (format!("file://{}", dir.path().display()), true),
    ] {
        without_a_key(&Database::open(&dsn).unwrap(), sealed);
    }
}

fn without_a_key(db: &Database, sealed: bool) {
    db.execute("CREATE TABLE n (k INTEGER, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO n VALUES (1, 10), (2, 10), (3, 30)", ())
        .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    let other = db.clone();
    let fired = Rc::new(Cell::new(false));
    let hook = Rc::clone(&fired);
    after_delete_rows_scanned(move || {
        other
            .execute("UPDATE n SET v = 20 WHERE k = 1", ())
            .unwrap();
        hook.set(true);
    });
    assert_eq!(db.execute("DELETE FROM n WHERE v + 0 = 10", ()).unwrap(), 1);
    assert!(fired.get(), "the commit did not come between");
    let left: Vec<(i64, i64)> = db
        .query("SELECT k, v FROM n ORDER BY k", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get(1).unwrap())
        })
        .collect();
    assert_eq!(left, [(1, 20), (3, 30)]);
}

/// Set once the rows are found: calls from then on are the judge's
static JUDGING: AtomicBool = AtomicBool::new(false);
static JUDGED: AtomicUsize = AtomicUsize::new(0);
static JUDGED_UNDER_FENCE: AtomicUsize = AtomicUsize::new(0);

/// Returns its argument and counts the calls the judge makes, and those
/// made under a seal fence
#[derive(Default)]
struct DeleteJudgedFence;

impl ScalarFunction for DeleteJudgedFence {
    fn name(&self) -> &str {
        "DELETE_JUDGED_FENCE"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DELETE_JUDGED_FENCE",
            FunctionType::Scalar,
            "test probe",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> stoolap::core::Result<Value> {
        if JUDGING.load(Ordering::Relaxed) {
            JUDGED.fetch_add(1, Ordering::Relaxed);
            if under_seal_fence() {
                JUDGED_UNDER_FENCE.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DeleteJudgedFence)
    }
}

#[test]
fn a_changed_row_is_judged_outside_the_seal_fence() {
    let _guard = serial();
    let _ = global_registry().register_scalar::<DeleteJudgedFence>();
    each_layout("outside_fence", &[], |db| {
        JUDGED.store(0, Ordering::Relaxed);
        JUDGED_UNDER_FENCE.store(0, Ordering::Relaxed);
        let other = db.clone();
        after_delete_rows_scanned(move || {
            other
                .execute("UPDATE t SET s = 'z' WHERE id = 1", ())
                .unwrap();
            JUDGING.store(true, Ordering::Relaxed);
        });
        let returned: Vec<i64> = db
            .query(
                "DELETE FROM t WHERE DELETE_JUDGED_FENCE(v) = 10 RETURNING id",
                (),
            )
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        JUDGING.store(false, Ordering::Relaxed);
        assert_eq!(returned, [1, 2]);
        // (calls the judge made, those under the seal fence): the changed row only, outside it
        let calls = (
            JUDGED.load(Ordering::Relaxed),
            JUDGED_UNDER_FENCE.load(Ordering::Relaxed),
        );
        assert_eq!(calls, (1, 0));
    });
}

#[test]
fn a_changed_row_is_judged_with_its_row_id_without_a_returning() {
    let dir = tempfile::tempdir().unwrap();
    for (dsn, sealed) in [
        ("memory://delete_judged_row_id".to_string(), false),
        (format!("file://{}", dir.path().display()), true),
    ] {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE n (k INTEGER, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO n VALUES (1, 10), (2, 10), (3, 30)", ())
            .unwrap();
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let fired = commit_between(&db, "UPDATE n SET v = 20 WHERE k = 1");
        let deleted = db.execute("DELETE FROM n WHERE _rowid_ + 0 = 1", ());
        assert!(fired.get(), "{dsn}: the commit did not come between");
        assert_eq!(deleted.unwrap_or_else(|e| panic!("{dsn}: {e}")), 1);
        let left: Vec<i64> = db
            .query("SELECT k FROM n ORDER BY k", ())
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(left, [2, 3], "{dsn}");
    }
}

#[test]
fn a_changed_row_is_judged_before_any_row_of_the_delete_goes() {
    each_layout("own_deletes", &[], |db| {
        db.execute("BEGIN", ()).unwrap();
        let fired = commit_between(db, "UPDATE t SET s = 'z' WHERE id = 1");
        let returned: Vec<i64> = db
            .query(
                "DELETE FROM t WHERE id = 2 OR v = (SELECT q.v FROM t q WHERE q.id = t.id + 1) RETURNING id",
                (),
            )
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        db.execute("COMMIT", ()).unwrap();
        assert!(fired.get(), "the commit did not come between");
        assert_eq!(returned, [1, 2]);
        assert_eq!(rows(db), [(3, 30, "c".into())]);
    });
}

#[test]
fn a_sealed_row_a_commit_in_flight_elsewhere_sealed_over_is_not_judged() {
    let _guard = serial();
    let _ = global_registry().register_scalar::<DeleteJudgedFence>();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 10, 'a'), (2, 10, 'b'), (3, 30, 'c')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // A commit into another table holds its sequence while `t` is sealed
    let (entered, in_flight) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let other = db.clone();
    let holder = std::thread::spawn(move || {
        after_commit_counted(move || {
            entered.send(()).unwrap();
            released.recv().unwrap();
        });
        other.execute("INSERT INTO u VALUES (1)", ()).unwrap();
    });
    in_flight.recv().unwrap();
    db.engine().seal_for_test().unwrap();
    JUDGED.store(0, Ordering::Relaxed);
    after_delete_rows_scanned(|| JUDGING.store(true, Ordering::Relaxed));
    let returned: Result<Vec<i64>, String> = db
        .query(
            "DELETE FROM t WHERE DELETE_JUDGED_FENCE(v) = 10 RETURNING id",
            (),
        )
        .map_err(|e| e.to_string())
        .and_then(|rows| {
            rows.map(|row| {
                row.map(|row| row.get(0).unwrap())
                    .map_err(|e| e.to_string())
            })
            .collect()
        });
    JUDGING.store(false, Ordering::Relaxed);
    release.send(()).unwrap();
    holder.join().unwrap();
    assert_eq!(returned, Ok(vec![1, 2]));
    assert_eq!(JUDGED.load(Ordering::Relaxed), 0, "no row was asked again");
}
