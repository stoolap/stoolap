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

//! A call's arguments are read as the function it names reads them: a
//! window function's default only for the rows of the partitions the
//! statement computes

use std::sync::atomic::{AtomicI64, Ordering};

use stoolap::core::{Result, Value};
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

static TICKS: AtomicI64 = AtomicI64::new(0);

/// 1 on its first call, then 2, 3 and on; registered as answering anew per call
#[derive(Default)]
struct Tick;

impl ScalarFunction for Tick {
    fn name(&self) -> &str {
        "TICK"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TICK",
            FunctionType::Scalar,
            "test counter",
            FunctionSignature::new(FunctionDataType::Integer, vec![], 0, 0),
        )
        .non_deterministic()
    }
    fn evaluate(&self, _args: &[Value]) -> Result<Value> {
        Ok(Value::Integer(TICKS.fetch_add(1, Ordering::Relaxed) + 1))
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Tick)
    }
}

const TABLE: [&str; 2] = [
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, g INTEGER)",
    "INSERT INTO t VALUES (1, 10, 1), (2, 20, 2), (3, 30, 3)",
];

#[test]
fn a_default_is_read_only_for_the_partitions_a_limit_keeps() {
    global_registry().register_scalar::<Tick>().unwrap();
    let db = Database::open("memory://window_default_limit").unwrap();
    for sql in TABLE {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT LAG(a, 1, TICK()) OVER (PARTITION BY g) FROM t LIMIT 1";
    let before = TICKS.load(Ordering::Relaxed);
    let rows = db.query(sql, ()).unwrap().count();
    let calls = TICKS.load(Ordering::Relaxed) - before;
    assert_eq!(rows, 1, "{sql}");
    assert_eq!(calls, 1, "{sql} read the default of rows its LIMIT drops");
}
