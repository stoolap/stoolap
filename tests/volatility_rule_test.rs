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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use stoolap::core::{Result, Value};
use stoolap::executor::Executor;
use stoolap::functions::registry::FunctionRegistry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::storage::mvcc::engine::MVCCEngine;

// The counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());
static CALLS: AtomicUsize = AtomicUsize::new(0);

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Returns its argument and counts its calls; registered as fixed
#[derive(Default)]
struct Counted;

impl ScalarFunction for Counted {
    fn name(&self) -> &str {
        "COUNTED"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "COUNTED",
            FunctionType::Scalar,
            "test fixed",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Counted)
    }
}

fn executor() -> Executor {
    let registry = FunctionRegistry::new();
    registry.register_scalar::<Counted>();
    let mut engine = MVCCEngine::in_memory();
    stoolap::storage::traits::Engine::open(&mut engine).unwrap();
    let executor = Executor::with_function_registry(Arc::new(engine), Arc::new(registry));
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO m VALUES (1, 1, 1), (2, 1, 2), (3, 1, 3), (4, 1, 4), (5, 1, 5)",
    ] {
        executor.execute(sql).unwrap();
    }
    executor
}

/// The rows of `sql` and the calls of COUNTED it made
fn run(executor: &Executor, sql: &str) -> (Vec<Option<Value>>, usize) {
    let before = CALLS.load(Ordering::Relaxed);
    let mut rows = executor.execute(sql).unwrap();
    let mut found = Vec::new();
    while rows.next() {
        found.push(rows.row().get(0).cloned());
    }
    (found, CALLS.load(Ordering::Relaxed) - before)
}

#[test]
fn test_a_clock_stays_one_per_statement_beside_a_registered_function() {
    let _guard = serial();
    let executor = executor();
    let (found, calls) = run(
        &executor,
        "SELECT id FROM t WHERE id = 1 AND a > (SELECT v FROM m WHERE m.k = t.id \
         AND m.v = COUNTED(t.id) + 2 + 0 * LENGTH(CAST(NOW() AS TEXT)))",
    );
    assert_eq!(found, vec![Some(Value::Integer(1))]);
    assert_eq!(
        calls, 1,
        "NOW is one value per statement whatever else the registry holds, so the \
         value is computed once per outer row, not per inner row"
    );
}

#[test]
fn test_a_fixed_registered_function_folds_by_the_registry_that_runs_it() {
    let _guard = serial();
    let executor = executor();
    let (found, calls) = run(
        &executor,
        "SELECT id FROM t WHERE id = 1 AND a > (SELECT v FROM m WHERE m.k = t.id \
         AND m.v = COUNTED(1) + m.k)",
    );
    assert_eq!(found, vec![Some(Value::Integer(1))]);
    assert!(
        calls < 5,
        "COUNTED is fixed in the registry that compiles it, so a call of constants \
         folds into the program instead of running per inner row: {calls} calls"
    );
}
