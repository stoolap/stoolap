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

//! A window aggregate's argument after the first is read once per run of
//! the statement, never while it is prepared or explained. The only test of
//! this binary: it registers a function process-wide

use std::sync::atomic::{AtomicI64, Ordering};

use stoolap::core::{Result, Value};
use stoolap::functions::scalar::SleepFunction;
use stoolap::functions::{global_registry, FunctionInfo, ScalarFunction};
use stoolap::Database;

static CALLS: AtomicI64 = AtomicI64::new(0);

#[derive(Default)]
struct CountedSeparator;

impl ScalarFunction for CountedSeparator {
    fn name(&self) -> &str {
        "COUNTED_SEPARATOR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo {
            name: "COUNTED_SEPARATOR".into(),
            ..SleepFunction.info()
        }
        .per_statement()
    }

    fn evaluate(&self, _args: &[Value]) -> Result<Value> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(Value::text(","))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Self)
    }
}

#[test]
fn a_window_argument_is_read_once_per_run_and_never_by_explain() {
    let db = Database::open("memory://window_argument_once").unwrap();
    global_registry().register_scalar::<CountedSeparator>();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    let sql = "SELECT STRING_AGG(CAST(a AS TEXT), COUNTED_SEPARATOR()) OVER () FROM t";
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .for_each(drop);
    assert_eq!(
        CALLS.load(Ordering::Relaxed),
        0,
        "EXPLAIN read the argument"
    );
    let rows: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(rows, vec!["1,2,3"; 3]);
    assert_eq!(CALLS.load(Ordering::Relaxed), 1, "the first run read it");
}
