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
//! scalar function's for each row whatever its name, a window function's
//! default only for the rows of the partitions the statement computes

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use stoolap::core::{Result, Value};
use stoolap::executor::Executor;
use stoolap::functions::registry::{global_registry, FunctionRegistry};
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::storage::mvcc::engine::MVCCEngine;
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

/// A scalar function named as a window function, returning its argument
#[derive(Default)]
struct ScalarNtile;

impl ScalarFunction for ScalarNtile {
    fn name(&self) -> &str {
        "NTILE"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "NTILE",
            FunctionType::Scalar,
            "test identity",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ScalarNtile)
    }
}

const TABLE: [&str; 2] = [
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, g INTEGER)",
    "INSERT INTO t VALUES (1, 10, 1), (2, 20, 2), (3, 30, 3)",
];

#[test]
fn a_scalar_function_named_as_a_window_function_reads_its_argument_for_each_row() {
    let registry = FunctionRegistry::new();
    registry.register_scalar::<ScalarNtile>();
    let mut engine = MVCCEngine::in_memory();
    stoolap::storage::traits::Engine::open(&mut engine).unwrap();
    let executor = Executor::with_function_registry(Arc::new(engine), Arc::new(registry));
    for sql in TABLE {
        executor.execute(sql).unwrap();
    }
    let sql = "SELECT NTILE((SELECT t.a)) FROM t ORDER BY t.id";
    let mut rows = executor
        .execute(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut got = Vec::new();
    while rows.next() {
        got.push(rows.row().get(0).cloned());
    }
    let want: Vec<_> = [10, 20, 30].map(|a| Some(Value::Integer(a))).into();
    assert_eq!(got, want, "{sql}");
}

#[test]
fn a_default_is_read_only_for_the_partitions_a_limit_keeps() {
    global_registry().register_scalar::<Tick>();
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
