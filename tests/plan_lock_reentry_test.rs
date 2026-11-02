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

//! A function a held plan's answer calls may change the table and run the
//! same prepared statement again: the run holds no lock on its plan while
//! it answers. The only test of this binary: it registers a function
//! process-wide

use std::cell::RefCell;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use stoolap::core::{Result, Value};
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

thread_local! {
    static REENTER: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct Reenter;

impl ScalarFunction for Reenter {
    fn name(&self) -> &str {
        "PLAN_LOCK_REENTER"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "PLAN_LOCK_REENTER",
            FunctionType::Scalar,
            "runs the armed statement once",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
        .non_deterministic()
    }
    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if let Some(run) = REENTER.with(|hook| hook.borrow_mut().take()) {
            run();
        }
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Reenter)
    }
}

#[test]
fn a_function_in_a_held_answer_may_rerun_its_statement_after_ddl() {
    global_registry().register_scalar::<Reenter>();
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        let db = Database::open("memory://plan_lock_reentry").unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        let stmt = db
            .prepare("SELECT PLAN_LOCK_REENTER(COUNT(*)) FROM t")
            .unwrap();
        // The first run plans it, the second answers from the table's count
        for _ in 0..2 {
            stmt.query(()).unwrap().for_each(drop);
        }
        let (other, again) = (db.clone(), stmt.clone());
        REENTER.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                other
                    .execute("ALTER TABLE t ADD COLUMN z INTEGER", ())
                    .unwrap();
                let rows: Vec<i64> = again
                    .query(())
                    .unwrap()
                    .map(|row| row.unwrap().get(0).unwrap())
                    .collect();
                assert_eq!(rows, [2]);
            }))
        });
        let rows: Vec<i64> = stmt
            .query(())
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        assert_eq!(rows, [2]);
        let _ = done.send(());
    });
    assert!(
        finished.recv_timeout(Duration::from_secs(20)).is_ok(),
        "the statement waited on its own plan's lock"
    );
}
