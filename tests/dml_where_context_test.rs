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

//! The pre-compiled memory-WHERE path must carry the execution context's
//! transaction id, matching the evaluator path it replaced.

use stoolap::api::Database;

#[test]
fn update_memory_where_sees_transaction_id() {
    let db = Database::open("memory://dmlctx_upd").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();

    let mut tx = db.begin().unwrap();
    // CURRENT_TRANSACTION_ID() is not pushable, forcing the memory filter
    let n = tx
        .execute(
            "UPDATE t SET a = 1 WHERE id + 0 = 1 AND CURRENT_TRANSACTION_ID() > 0",
            (),
        )
        .unwrap();
    assert_eq!(n, 1, "WHERE must see the transaction id, not NULL");
    tx.commit().unwrap();
    let a: i64 = db.query_one("SELECT a FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(a, 1);
}

#[test]
fn delete_memory_where_sees_transaction_id() {
    let db = Database::open("memory://dmlctx_del").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();

    let mut tx = db.begin().unwrap();
    let n = tx
        .execute(
            "DELETE FROM t WHERE id + 0 = 1 AND CURRENT_TRANSACTION_ID() > 0",
            (),
        )
        .unwrap();
    assert_eq!(n, 1);
    tx.commit().unwrap();
    let c: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(c, 0);
}

mod registry_override {
    use std::sync::Arc;
    use stoolap::core::{Result, Value};
    use stoolap::executor::Executor;
    use stoolap::functions::registry::FunctionRegistry;
    use stoolap::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
    };
    use stoolap::storage::mvcc::engine::MVCCEngine;

    /// ABS override that always returns -1: distinguishes the executor's
    /// registry from the global one
    #[derive(Default)]
    struct AbsAlwaysNeg;

    impl ScalarFunction for AbsAlwaysNeg {
        fn name(&self) -> &str {
            "ABS"
        }
        fn info(&self) -> FunctionInfo {
            FunctionInfo::new(
                "ABS",
                FunctionType::Scalar,
                "test override",
                FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
            )
        }
        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            let _ = args;
            Ok(Value::Integer(-1))
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(AbsAlwaysNeg)
        }
    }

    fn count(executor: &Executor, sql: &str) -> i64 {
        let mut r = executor.execute(sql).unwrap();
        assert!(r.next());
        match r.row().get(0) {
            Some(Value::Integer(n)) => *n,
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn residual_where_uses_the_executors_registry() {
        let mut engine = MVCCEngine::in_memory();
        stoolap::storage::traits::Engine::open(&mut engine).unwrap();
        let engine = Arc::new(engine);
        let registry = FunctionRegistry::new();
        registry.register_scalar::<AbsAlwaysNeg>();
        let executor = Executor::with_function_registry(engine, Arc::new(registry));

        executor
            .execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)")
            .unwrap();
        executor.execute("INSERT INTO t VALUES (1, 10)").unwrap();

        // The row-selection phase runs with builtin ABS (ABS(10) = 10,
        // row selected); the residual re-check in the setter must use the
        // EXECUTOR's registry, whose ABS returns -1 and rejects the row.
        // Matching the pre-existing evaluator behavior means: no update.
        executor
            .execute("UPDATE t SET a = 1 WHERE id + 0 = 1 AND ABS(a) = 10")
            .unwrap();

        assert_eq!(
            count(&executor, "SELECT COUNT(*) FROM t WHERE a = 1"),
            0,
            "the residual re-check must run the executor's ABS override"
        );
        assert_eq!(count(&executor, "SELECT COUNT(*) FROM t WHERE a = 10"), 1);
    }
}
