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

//! A WHERE conjunct the scan's filter takes is not asked again of the rows
//! it passed, and one that can fail does not run ahead of a conjunct the
//! filter leaves out

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::functions::{
    global_registry, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
    ScalarFunction,
};
use stoolap::Database;

static CALLS: AtomicUsize = AtomicUsize::new(0);
static COUNTED: Mutex<()> = Mutex::new(());

/// LENGTH, counting its calls
#[derive(Default)]
struct CountedLength;

impl ScalarFunction for CountedLength {
    fn name(&self) -> &str {
        "PUSHED_ONCE_LENGTH"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "PUSHED_ONCE_LENGTH",
            FunctionType::Scalar,
            "test length",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }
    fn evaluate(&self, args: &[Value]) -> stoolap::core::Result<Value> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        Ok(match args.first() {
            Some(Value::Text(s)) => Value::Integer(s.len() as i64),
            _ => Value::null_unknown(),
        })
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CountedLength)
    }
}

const ROWS: usize = 20;

fn ids(db: &Database, sql: &str, prepared: bool) -> Result<Vec<i64>, String> {
    let rows = match prepared {
        true => db.prepare(sql).and_then(|stmt| stmt.query(())),
        false => db.query(sql, ()),
    };
    rows.and_then(|rows| rows.map(|row| row?.get::<i64>(0)).collect())
        .map_err(|e| e.to_string())
}

#[test]
fn a_conjunct_the_scan_takes_runs_once_a_row() {
    let _turn = COUNTED.lock().unwrap_or_else(|e| e.into_inner());
    global_registry()
        .register_scalar::<CountedLength>()
        .unwrap();
    let db = Database::open("memory://pushed_conjunct_once_runs").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    for id in 1..=ROWS as i64 {
        let s = if id % 2 == 0 { "ab" } else { "ba" };
        db.execute("INSERT INTO t VALUES ($1, $2)", (id, s))
            .unwrap();
    }
    let sql = "SELECT id FROM t WHERE PUSHED_ONCE_LENGTH(s) > 1 AND UPPER(s) LIKE 'A%'";
    for prepared in [false, true] {
        CALLS.store(0, Ordering::Relaxed);
        let even: Vec<i64> = (1..=ROWS as i64).filter(|id| id % 2 == 0).collect();
        assert_eq!(ids(&db, sql, prepared), Ok(even), "prepared {prepared}");
        assert_eq!(
            CALLS.load(Ordering::Relaxed),
            ROWS,
            "one call a row, prepared {prepared}"
        );
    }
}

#[test]
fn a_fallible_conjunct_waits_for_one_the_scan_leaves_out() {
    let db = Database::open("memory://pushed_conjunct_once_waits").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, p TEXT, x INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a', $1)", (i64::MIN,))
        .unwrap();
    let sql = "SELECT id FROM t WHERE UPPER(p) LIKE 'Q%' AND ABS(x) >= 0";
    for prepared in [false, true] {
        assert_eq!(ids(&db, sql, prepared), Ok(vec![]), "prepared {prepared}");
    }
}
