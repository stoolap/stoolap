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

//! A value a WHERE compares a column with fails only for a row that reaches
//! it, whatever runs the statement and wherever its rows are

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use stoolap::core::Value;
use stoolap::Database;

/// Each WHERE with whether a row reaches its failing value, `$1` being 'a'
const SHAPES: [(&str, bool); 16] = [
    ("FALSE AND v IN (ABS($1))", false),
    ("FALSE AND v IN (ABS($1)) AND EXISTS (SELECT 1)", false),
    ("FALSE AND v BETWEEN ABS($1) AND 3", false),
    ("id > 100 AND v = ABS($1)", false),
    ("id > 100 AND v = ABS($1) AND EXISTS (SELECT 1)", false),
    ("v > 0 OR v = ABS($1)", false),
    ("v > 0 OR v IN (ABS($1), 2)", false),
    ("id > 100 AND v = (SELECT w FROM u)", false),
    ("id > 100 AND v IN ((SELECT w FROM u), 2)", false),
    ("v > 0 OR v = (SELECT w FROM u)", false),
    ("v = ABS($1)", true),
    ("v = ABS($1) AND EXISTS (SELECT 1)", true),
    ("v IN (ABS($1)) AND EXISTS (SELECT 1)", true),
    ("v < 0 OR v = ABS($1)", true),
    ("v = (SELECT w FROM u)", true),
    ("id = ABS($1) AND EXISTS (SELECT 1)", true),
];

/// Each statement shape the WHERE is asked in, and whether it writes
const SITES: [(&str, bool); 7] = [
    ("SELECT id FROM t WHERE {w}", false),
    ("SELECT id FROM t WHERE {w} ORDER BY v LIMIT 2", false),
    ("SELECT COUNT(*) FROM t WHERE {w}", false),
    ("SELECT v, COUNT(*) FROM t WHERE {w} GROUP BY v", false),
    ("SELECT t.id FROM t JOIN u ON u.k = t.id WHERE {w}", false),
    ("DELETE FROM t WHERE {w}", true),
    ("UPDATE t SET v = v WHERE {w}", true),
];

fn tables(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "CREATE INDEX t_v ON t(v)",
        "CREATE TABLE u (k INTEGER PRIMARY KEY, w INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    refill(&db, sealed);
    db
}

/// `t`'s rows as each statement starts from them
fn refill(db: &Database, sealed: bool) {
    db.execute("DELETE FROM t", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
}

#[test]
fn a_where_value_fails_only_for_a_row_that_reaches_it() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = match sealed {
            true => format!("file://{}", dir.path().display()),
            false => "memory://where_value_error_hot".to_string(),
        };
        let db = tables(&dsn, sealed);
        for (site, writes) in SITES {
            for (where_, reached) in SHAPES {
                let sql = site.replace("{w}", where_);
                // The second run reads the plan the first one kept
                for run in 0..2 {
                    let answered = match writes {
                        true => {
                            let answered = db.execute(&sql, ("a",)).is_ok();
                            refill(&db, sealed);
                            answered
                        }
                        false => db
                            .query(&sql, ("a",))
                            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
                            .is_ok(),
                    };
                    if answered == reached {
                        wrong.push(format!("{sql} (sealed {sealed}, run {run})"));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "these answered where a row reached the failing value, or failed where none did:\n{}",
        wrong.join("\n")
    );
}

static TICKS: AtomicI64 = AtomicI64::new(0);

/// 1 on its first call, then 2, 3 and on; registered as answering anew per call
#[derive(Default)]
struct Tick;

impl stoolap::functions::ScalarFunction for Tick {
    fn name(&self) -> &str {
        "TICK"
    }
    fn info(&self) -> stoolap::functions::FunctionInfo {
        use stoolap::functions::{FunctionDataType, FunctionInfo, FunctionSignature, FunctionType};
        FunctionInfo::new(
            "TICK",
            FunctionType::Scalar,
            "test counter",
            FunctionSignature::new(FunctionDataType::Integer, vec![], 0, 0),
        )
        .non_deterministic()
    }
    fn evaluate(&self, _args: &[Value]) -> stoolap::core::Result<Value> {
        Ok(Value::Integer(TICKS.fetch_add(1, Ordering::Relaxed) + 1))
    }
    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(Tick)
    }
}

#[test]
fn a_where_value_is_computed_once_when_another_fails() {
    let registry = stoolap::functions::registry::FunctionRegistry::new();
    registry.register_scalar::<Tick>();
    let mut engine = stoolap::storage::mvcc::engine::MVCCEngine::in_memory();
    stoolap::storage::traits::Engine::open(&mut engine).unwrap();
    let executor =
        stoolap::executor::Executor::with_function_registry(Arc::new(engine), Arc::new(registry));
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 1)",
    ] {
        executor.execute(sql).unwrap();
    }
    let params: stoolap::api::ParamVec = std::iter::once(Value::text("a")).collect();
    let answer = executor
        .execute_with_params(
            "SELECT id FROM t WHERE v = (SELECT TICK()) AND v = ABS($1)",
            params,
        )
        .map(|mut rows| rows.next());
    assert!(
        answer.is_err(),
        "the row the first value let through reached the failing one, got {answer:?}"
    );
    assert_eq!(
        TICKS.load(Ordering::Relaxed),
        1,
        "the subquery's value was computed again"
    );
}
