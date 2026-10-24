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

//! A DISTINCT ON key the SELECT list computes is read from that item, not
//! computed again. The only test of this binary: it registers a function
//! process-wide

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
fn a_key_the_select_list_computes_runs_once_per_row() {
    global_registry().register_scalar::<NextCount>();
    let db = Database::open("memory://distinct_on_key_volatile").unwrap();
    stoolap::Executor::new(std::sync::Arc::clone(db.engine()))
        .function_registry()
        .register_scalar::<NextCount>();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    let got: Vec<i64> = db
        .query(
            "SELECT DISTINCT ON (NEXT_COUNT()) NEXT_COUNT() AS k, COUNT(*) FROM t \
             GROUP BY a ORDER BY a",
            (),
        )
        .unwrap()
        .map(|row| match row.unwrap().get_value(0).unwrap() {
            Value::Integer(k) => *k,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(got, vec![1, 2, 3], "one call per group");
    let before = CALLS.load(Ordering::Relaxed);
    let rows = db
        .query(
            "SELECT DISTINCT ON (NEXT_COUNT()) *, NEXT_COUNT() AS \"NEXT_COUNT()\", \
             ROW_NUMBER() OVER (ORDER BY id) FROM t",
            (),
        )
        .unwrap()
        .count();
    assert_eq!(rows, 3);
    assert_eq!(
        CALLS.load(Ordering::Relaxed) - before,
        3,
        "beside a star, one call per row"
    );
    for order_by in ["ORDER BY id", ""] {
        let before = CALLS.load(Ordering::Relaxed);
        let sql = format!(
            "SELECT DISTINCT ON (NEXT_COUNT()) t.*, NEXT_COUNT() AS k, t.*, \
             ROW_NUMBER() OVER (ORDER BY id) FROM t {order_by}"
        );
        assert_eq!(db.query(&sql, ()).unwrap().count(), 3, "{sql}");
        assert_eq!(
            CALLS.load(Ordering::Relaxed) - before,
            3,
            "between two stars, one call per row: {sql}"
        );
    }
}
