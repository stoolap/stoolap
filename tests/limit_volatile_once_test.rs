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

//! A LIMIT or OFFSET whose call draws a new value each time is read once,
//! before the statement reads a row, whichever path reads the rows. The
//! only test of this binary: it registers a function process-wide

use std::sync::atomic::{AtomicI64, Ordering};

use stoolap::core::{Result, Value};
use stoolap::functions::scalar::SleepFunction;
use stoolap::functions::{global_registry, FunctionInfo, ScalarFunction};
use stoolap::Database;

static CALLS: AtomicI64 = AtomicI64::new(0);

#[derive(Default)]
struct NextCount;

impl ScalarFunction for NextCount {
    fn name(&self) -> &str {
        "NEXT_COUNT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo {
            name: "NEXT_COUNT".into(),
            ..SleepFunction.info()
        }
    }

    fn evaluate(&self, _args: &[Value]) -> Result<Value> {
        Ok(Value::Integer(CALLS.fetch_add(1, Ordering::Relaxed) + 1))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Self)
    }
}

/// A database holding `t`, its view and an index, with NEXT_COUNT known
fn open(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    stoolap::Executor::new(std::sync::Arc::clone(db.engine()))
        .function_registry()
        .register_scalar::<NextCount>();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "INSERT INTO t SELECT value, value % 9, 'b' || value FROM generate_series(1, 200)",
        "CREATE INDEX t_a ON t (a)",
        "CREATE VIEW v AS SELECT id, a, b FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    for shape in [
        "SELECT id + 0 AS k FROM t ORDER BY k",
        "SELECT id, a FROM t ORDER BY a + 0, id",
        "SELECT id FROM t ORDER BY b, id",
        "SELECT id, a FROM t ORDER BY id",
        "SELECT id, a FROM t ORDER BY a",
        "SELECT id FROM t",
        "SELECT id + 0 AS k FROM (SELECT id FROM t) s ORDER BY k",
        "SELECT id FROM v ORDER BY a + 0, id",
        "SELECT t.id FROM t JOIN t u ON u.id = t.id ORDER BY t.id + 0",
    ] {
        for (tail, reads) in [
            (" LIMIT NEXT_COUNT()", 1),
            (" LIMIT NEXT_COUNT() OFFSET NEXT_COUNT()", 2),
        ] {
            let sql = format!("{shape}{tail}");
            let first = CALLS.load(Ordering::Relaxed) + 1;
            let rows = db.query(&sql, ()).unwrap().count() as i64;
            let calls = CALLS.load(Ordering::Relaxed) - first + 1;
            if calls != reads || rows != first {
                wrong.push(format!(
                    "{phase}: {sql}: {calls} calls, {rows} rows, LIMIT read as {first}"
                ));
            }
        }
    }
}

#[test]
fn a_volatile_limit_and_offset_are_read_once_on_every_path() {
    global_registry().register_scalar::<NextCount>();
    let mut wrong = Vec::new();
    let db = open("memory://limit_volatile_once");
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let db = open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ));
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 4 WHERE id % 30 = 0", ())
        .unwrap();
    check(&db, "sealed with hot updates", &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
