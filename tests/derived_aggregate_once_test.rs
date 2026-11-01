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

//! A statement that aggregates a derived table runs the derived table's
//! statement once, whichever way its rows are aggregated

use std::sync::atomic::{AtomicI64, Ordering};

use stoolap::core::{Result, Value};
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

static TICKS: AtomicI64 = AtomicI64::new(0);

/// 1 on its first call, then 2, 3 and on; registered as answering anew per
/// call, by a name no other test uses: a derived table's statement reads
/// the process-wide registry
#[derive(Default)]
struct Tick;

impl ScalarFunction for Tick {
    fn name(&self) -> &str {
        "DERIVED_ONCE_TICK"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DERIVED_ONCE_TICK",
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

/// Each statement over the ten rows of t, with the answer its one run of
/// the derived table gives: ticks 1 to 10
const STATEMENTS: &[(&str, &str)] = &[
    (
        "SELECT COUNT(*), SUM(c) FROM \
         (SELECT id, DERIVED_ONCE_TICK() AS c FROM t ORDER BY c) s",
        "10 55",
    ),
    (
        "SELECT k, SUM(c) FROM (SELECT id, k, DERIVED_ONCE_TICK() AS c FROM t ORDER BY c) s \
         GROUP BY k HAVING COUNT(*) > 0 ORDER BY k",
        "0 30 1 25",
    ),
    (
        "SELECT k, SUM(c) FROM (SELECT id, k, DERIVED_ONCE_TICK() AS c FROM t ORDER BY c) s \
         GROUP BY k ORDER BY k",
        "0 30 1 25",
    ),
];

#[test]
fn an_aggregate_over_a_derived_table_runs_its_statement_once() {
    global_registry().register_scalar::<Tick>();
    let db = Database::open("memory://derived_aggregate_once").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO t SELECT value, value % 2 FROM generate_series(1, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, want) in STATEMENTS {
        TICKS.store(0, Ordering::Relaxed);
        let got: Vec<String> = db
            .query(sql, ())
            .unwrap()
            .flat_map(|row| {
                let row = row.unwrap();
                (0..row.len())
                    .map(|i| row.get::<i64>(i).unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        let got = got.join(" ");
        let ticks = TICKS.load(Ordering::Relaxed);
        if got != *want || ticks != 10 {
            wrong.push(format!(
                "{sql}: {got} after {ticks} ticks, want {want} after 10"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
