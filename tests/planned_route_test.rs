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

//! A statement the planner planned runs on the runner when its key read
//! declines a run, never on the general path

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::core::Value;
use stoolap::test_failpoints::after_prepared_plan_taken;
use stoolap::Database;

#[test]
fn a_declined_key_read_runs_on_the_runner() {
    let db = Database::open("memory://planned_route_declined").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let statement = db.prepare("SELECT id + 0, v FROM t WHERE id = $1").unwrap();
    let run = |key: Value| {
        let ran = Rc::new(Cell::new(false));
        let seen = Rc::clone(&ran);
        after_prepared_plan_taken(move || seen.set(true));
        let rows = statement.query((key,)).unwrap().count();
        (rows, ran.get())
    };
    assert_eq!(run(Value::Integer(1)), (1, false), "the key read answers");
    assert_eq!(
        run(Value::Float(1.5)),
        (0, true),
        "a key the read declines is the runner's, not the general path's"
    );
}
