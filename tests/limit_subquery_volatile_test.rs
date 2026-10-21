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

//! A LIMIT and an OFFSET spelled alike keep their own counts, also in a
//! statement the executor rebuilds after reading them. The only test of
//! this binary: it registers a function process-wide

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

#[test]
fn a_limit_and_an_offset_spelled_alike_keep_their_own_counts() {
    global_registry().register_scalar::<NextCount>();
    let db = Database::open("memory://limit_subquery_volatile").unwrap();
    stoolap::Executor::new(std::sync::Arc::clone(db.engine()))
        .function_registry()
        .register_scalar::<NextCount>();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3), (4)", ())
        .unwrap();
    let ids: Vec<i64> = db
        .query(
            "SELECT id, COUNT(*) + (SELECT 0) FROM t GROUP BY id ORDER BY id \
             LIMIT (SELECT NEXT_COUNT()) OFFSET (SELECT NEXT_COUNT())",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(CALLS.load(Ordering::Relaxed), 2, "each count read once");
    assert_eq!(ids, [3], "LIMIT 1 OFFSET 2");
}
