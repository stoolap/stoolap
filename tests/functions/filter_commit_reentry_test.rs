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

//! A function a read's filter calls may commit to the table through another
//! handle: the read asks its filter with none of the store's locks held

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
    static WRITER: RefCell<Option<Database>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct CommitsOnce;

impl ScalarFunction for CommitsOnce {
    fn name(&self) -> &str {
        "FILTER_COMMITS_ONCE"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "FILTER_COMMITS_ONCE",
            FunctionType::Scalar,
            "commits one row through the armed handle",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if let Some(db) = WRITER.with(|writer| writer.borrow_mut().take()) {
            db.execute("INSERT INTO t VALUES (100, 2, 1)", ())?;
        }
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CommitsOnce)
    }
}

#[test]
fn a_timed_index_read_asks_its_filter_outside_the_stores_locks() {
    global_registry().register_scalar::<CommitsOnce>().unwrap();
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        let db = Database::open("memory://filter_commit_reentry").unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 1, 1), (2, 1, 2), (3, 1, 3)", ())
            .unwrap();
        db.execute("CREATE INDEX t_g ON t (g)", ()).unwrap();
        db.execute("PRAGMA query_timeout = 3600000", ()).unwrap();
        WRITER.with(|writer| *writer.borrow_mut() = Some(db.clone()));
        let read = db
            .query(
                "SELECT id FROM t WHERE g > 0 AND FILTER_COMMITS_ONCE(v) > 0",
                (),
            )
            .map(|rows| rows.count());
        let _ = done.send(read.map_err(|e| e.to_string()));
    });
    let finished = finished.recv_timeout(Duration::from_secs(10));
    assert!(matches!(finished, Ok(Ok(3))), "the read gave {finished:?}");
}
