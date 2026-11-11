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

//! A DELETE asks its WHERE of a row once when nothing changed the row
//! between finding it and deleting it

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use stoolap::functions::{
    global_registry, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
    ScalarFunction,
};
use stoolap::{Database, Value};

static CALLS: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Returns its argument and counts its calls
#[derive(Default)]
struct DeleteWhereCounted;

impl ScalarFunction for DeleteWhereCounted {
    fn name(&self) -> &str {
        "DELETE_WHERE_COUNTED"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DELETE_WHERE_COUNTED",
            FunctionType::Scalar,
            "test counter",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> stoolap::core::Result<Value> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DeleteWhereCounted)
    }
}

fn table(name: &str) -> Database {
    filled(Database::open(&format!("memory://{name}")).unwrap())
}

fn filled(db: Database) -> Database {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 10), (3, 30)", ())
        .unwrap();
    db
}

/// `t` sealed into a volume of a file database in `dir`
fn sealed_table(dir: &std::path::Path) -> Database {
    let db = filled(Database::open(&format!("file://{}", dir.display())).unwrap());
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

#[test]
fn a_row_unchanged_since_it_was_found_is_asked_its_where_once() {
    let _guard = serial();
    let _ = global_registry().register_scalar::<DeleteWhereCounted>();
    let referenced = "DELETE FROM t WHERE DELETE_WHERE_COUNTED(v) = 10";
    let returning = "DELETE FROM t WHERE DELETE_WHERE_COUNTED(v) = 10 RETURNING id";
    let prepared = "DELETE FROM t WHERE DELETE_WHERE_COUNTED(v) = $1 RETURNING id";
    let mut got = Vec::new();
    for (case, sql, sealed) in [
        ("referenced", referenced, false),
        ("returning", returning, false),
        ("prepared", prepared, false),
        ("referenced sealed", referenced, true),
        ("returning sealed", returning, true),
        ("prepared sealed", prepared, true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => sealed_table(dir.path()),
            false => table(&format!("delete_where_asked_once_{case}")),
        };
        CALLS.store(0, Ordering::Relaxed);
        let deleted = match case.split(' ').next().unwrap() {
            "referenced" => {
                db.execute(
                    "CREATE TABLE c (id INTEGER PRIMARY KEY, t_id INTEGER REFERENCES t(id))",
                    (),
                )
                .unwrap();
                db.execute(sql, ()).unwrap() as usize
            }
            "returning" => db.query(sql, ()).unwrap().count(),
            _ => db.prepare(sql).unwrap().query((10,)).unwrap().count(),
        };
        got.push((case, deleted, CALLS.load(Ordering::Relaxed)));
    }
    // (case, rows deleted, calls): one call per row read
    assert_eq!(
        got,
        [
            ("referenced", 2, 3),
            ("returning", 2, 3),
            ("prepared", 2, 3),
            ("referenced sealed", 2, 3),
            ("returning sealed", 2, 3),
            ("prepared sealed", 2, 3)
        ]
    );
}

#[test]
fn a_row_this_transaction_wrote_is_asked_its_where_once() {
    let _guard = serial();
    let _ = global_registry().register_scalar::<DeleteWhereCounted>();
    let db = table("delete_where_asked_once_own_write");
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE t SET v = 10 WHERE id = 3", ()).unwrap();
    CALLS.store(0, Ordering::Relaxed);
    let deleted = db
        .query(
            "DELETE FROM t WHERE DELETE_WHERE_COUNTED(v) = 10 RETURNING id",
            (),
        )
        .unwrap()
        .count();
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(deleted, 3);
    assert_eq!(CALLS.load(Ordering::Relaxed), 3, "one call per row read");
}
