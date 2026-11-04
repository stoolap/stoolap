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

//! The HAVING of a grouped join a LIMIT reduces is compiled over the
//! group rows without the evaluator bridge

#![cfg(feature = "test-failpoints")]

use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::test_failpoints::bridge_compiles;
use stoolap::Database;

// The bridge counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

/// Each statement, its parameters and the rows it answers in some order,
/// every value written out: u 1 has three orders, u 2 one, u 3 two, u 4 none
const CASES: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT u.id, COUNT(o.id) FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id HAVING COUNT(o.id) > 1 LIMIT 5",
        &[],
        &["1|3", "3|2"],
    ),
    (
        "SELECT u.id, u.name, SUM(o.v) AS s FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id, u.name HAVING SUM(o.v) > $1 LIMIT 5",
        &[25],
        &["1|a|60", "3|c|50"],
    ),
    (
        "SELECT u.id, u.name, SUM(o.v) AS s FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id, u.name HAVING SUM(o.v) > $1 LIMIT 5",
        &[55],
        &["1|a|60"],
    ),
    (
        "SELECT u.id, MAX(o.v) FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id HAVING MAX(o.v) - MIN(o.v) >= 10 LIMIT 5",
        &[],
        &["1|30", "3|30"],
    ),
];

fn rows(db: &Database, sql: &str, params: &[i64]) -> Vec<String> {
    let params: Vec<Value> = params.iter().map(|&p| Value::Integer(p)).collect();
    let mut rows = db
        .query(sql, params)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|at| row.get::<Value>(at).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

/// u and o in a database of `name`'s own, o indexed on its join key
fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, name TEXT)",
        "INSERT INTO u VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')",
        "CREATE TABLE o (id INTEGER PRIMARY KEY, uid INTEGER, v INTEGER)",
        "CREATE INDEX o_uid ON o (uid)",
        "INSERT INTO o VALUES (1, 1, 10), (2, 1, 20), (3, 1, 30), (4, 2, 5), (5, 3, 20), (6, 3, 30)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn a_reduced_grouped_join_having_answers_as_written_without_the_bridge() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = open("grouped_index_join_having");
    let mut wrong = Vec::new();
    for (sql, params, expected) in CASES {
        for run in 0..2 {
            let before = bridge_compiles();
            let answer = rows(&db, sql, params);
            let compiled = bridge_compiles() - before;
            if answer != *expected {
                wrong.push(format!(
                    "{sql} {params:?}, run {run}: {answer:?}, written {expected:?}"
                ));
            }
            if compiled != 0 {
                wrong.push(format!(
                    "{sql} {params:?}, run {run}: {compiled} bridge compiles"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_pattern_parameter_in_a_branch_no_group_reaches_raises_nothing() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = open("grouped_index_join_having_pattern");
    let sql = "SELECT u.id, u.name, COUNT(o.id) FROM u JOIN o ON o.uid = u.id \
               GROUP BY u.id, u.name HAVING COUNT(o.id) > 0 OR u.name REGEXP $1 LIMIT 5";
    let params = vec![Value::text("[")];
    let got: Result<Vec<String>, String> = db
        .query(sql, params)
        .and_then(|rows| {
            rows.map(|row| row.map(|row| row.get::<Value>(0).unwrap().to_string()))
                .collect()
        })
        .map_err(|e| e.to_string());
    let mut got = got.unwrap_or_else(|e| panic!("{sql} with '[': {e}"));
    got.sort();
    assert_eq!(got, ["1", "2", "3"], "{sql}");
}

/// The database a HAVING's function reads again, through a handle of its own
static NESTED: Mutex<Option<Database>> = Mutex::new(None);

/// Runs a reduced grouped join with a HAVING before it returns its argument
#[derive(Default)]
struct Reenter;

impl ScalarFunction for Reenter {
    fn name(&self) -> &str {
        "GROUPED_HAVING_REENTER"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "GROUPED_HAVING_REENTER",
            FunctionType::Scalar,
            "runs a grouped join, then returns its argument",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> stoolap::core::Result<Value> {
        let nested = NESTED.lock().unwrap_or_else(|e| e.into_inner());
        let db = nested.as_ref().expect("the nested database is set");
        let inner = rows(
            db,
            "SELECT u.id, COUNT(o.id) FROM u JOIN o ON o.uid = u.id \
             GROUP BY u.id HAVING COUNT(o.id) > 1 LIMIT 5",
            &[],
        );
        assert_eq!(inner, ["1|3", "3|2"], "the inner join's groups");
        Ok(args[0].clone())
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Reenter)
    }
}

#[test]
fn a_having_whose_function_runs_such_a_join_again_answers() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    global_registry().register_scalar::<Reenter>().unwrap();
    let db = open("grouped_index_join_having_reenter");
    *NESTED.lock().unwrap_or_else(|e| e.into_inner()) = Some(db.clone());
    let got = rows(
        &db,
        "SELECT u.id, COUNT(o.id) FROM u JOIN o ON o.uid = u.id \
         GROUP BY u.id HAVING GROUPED_HAVING_REENTER(COUNT(o.id)) > 1 LIMIT 5",
        &[],
    );
    NESTED.lock().unwrap_or_else(|e| e.into_inner()).take();
    assert_eq!(got, ["1|3", "3|2"]);
}
