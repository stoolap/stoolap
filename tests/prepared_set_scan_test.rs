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

//! A top-level `column [NOT] IN (block)` or `column op ANY|ALL (block)` over
//! an uncorrelated block is read before the scan and taken by the scan: the
//! statement answers as the program does when the conjunct is asked of each
//! row (`OR id < 0`, FALSE for every row), for every member set and column
//! type, hot and sealed

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use stoolap::core::Value;
use stoolap::executor::Executor;
use stoolap::functions::{
    global_registry, FunctionDataType, FunctionInfo, FunctionRegistry, FunctionSignature,
    FunctionType, ScalarFunction,
};
use stoolap::storage::mvcc::engine::MVCCEngine;
use stoolap::test_failpoints::{prepared_strategy_runs, PreparedStrategy};
use stoolap::Database;

/// The strategy counts are process-wide: tests that read or move them take
/// turns
static COUNTED: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// t: id 1..40, i INTEGER (NULL every 9th), f FLOAT, s TEXT; m holds the
/// member sets, picked by `k`: 1 numbers, 2 numbers and a NULL, 3 NULL
/// only, 4 TEXT numbers, 5 no row at all
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, s TEXT)",
        "INSERT INTO t SELECT g.value, CASE WHEN g.value % 9 = 0 THEN NULL ELSE g.value % 11 END, \
         (g.value % 7) * 1.5, CAST(g.value % 5 AS TEXT) FROM generate_series(1, 40) g",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, w TEXT)",
        "INSERT INTO m VALUES (1, 1, 3, '3'), (2, 1, 7, '7'), (3, 2, 4, '4'), (4, 2, NULL, NULL), \
         (5, 3, NULL, NULL), (6, 4, 2, '2'), (7, 4, 9, '9')",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

const COLUMNS: &[&str] = &["i", "f", "s"];

const TESTS: &[&str] = &[
    "IN", "NOT IN", "= ANY", "<> ALL", "> ANY", "< ALL", ">= ALL", "<= ANY", "<> ANY", "= ALL",
];

const BLOCKS: &[&str] = &[
    "SELECT v FROM m WHERE k = 1",
    "SELECT v FROM m WHERE k = 2",
    "SELECT v FROM m WHERE k = 3",
    "SELECT w FROM m WHERE k = 4",
    "SELECT v FROM m WHERE k = 5",
];

fn ids(db: &Database, sql: &str) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(
            row.map_err(|e| e.to_string())?
                .get::<i64>(0)
                .map_err(|e| e.to_string())?,
        );
    }
    out.sort();
    Ok(out)
}

#[test]
fn a_scan_takes_a_set_conjunct_as_the_general_path_answers_it() {
    let _turn = COUNTED.lock().unwrap_or_else(|e| e.into_inner());
    let mut wrong = Vec::new();
    let mut taken = 0;
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://prepared_set_scan").unwrap();
            fill(&db);
            db
        };
        for column in COLUMNS {
            for test in TESTS {
                for block in BLOCKS {
                    let condition = format!("{column} {test} ({block})");
                    for shape in [
                        "SELECT id FROM {t} WHERE {c}",
                        "SELECT id FROM {t} WHERE id > 5 AND {c}",
                        "SELECT id FROM {t} WHERE {c} ORDER BY id LIMIT 4 OFFSET 2",
                    ] {
                        let plain = shape.replace("{c}", &condition).replace("{t}", "t");
                        let per_row = shape
                            .replace("{c}", &format!("({condition} OR id < 0)"))
                            .replace("{t}", "t");
                        let before = prepared_strategy_runs(PreparedStrategy::ScanTakesMembers);
                        let got = ids(&db, &plain);
                        taken +=
                            prepared_strategy_runs(PreparedStrategy::ScanTakesMembers) - before;
                        let want = ids(&db, &per_row);
                        if got != want {
                            wrong.push(format!(
                                "sealed {sealed}: {plain}: {got:?}, per row {want:?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    assert!(taken > 0, "no statement reached the set scan");
}

#[test]
fn a_block_error_belongs_to_the_row_that_needs_the_answer() {
    let _turn = COUNTED.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://prepared_set_scan_errors").unwrap();
    fill(&db);
    db.execute("CREATE TABLE e (id INTEGER PRIMARY KEY, i INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 9223372036854775807), (2, 1)", ())
        .unwrap();
    for block in [
        "(SELECT v * 9223372036854775807 FROM m WHERE k = 1)",
        "(SELECT SUM(v) FROM o)",
    ] {
        for test in ["IN", "NOT IN", "> ANY", ">= ALL", "= ANY", "<> ALL"] {
            let empty = format!("SELECT id FROM e WHERE i {test} {block}");
            assert_eq!(ids(&db, &empty), Ok(vec![]), "{empty}: no row needs it");
            let rows = format!("SELECT id FROM t WHERE i {test} {block}");
            let got = ids(&db, &rows);
            let per_row = ids(&db, &rows.replace("WHERE i", "WHERE id < 0 OR i"));
            assert!(
                got.as_ref().is_err_and(|e| e.contains("overflow")),
                "{rows}: {got:?}"
            );
            assert_eq!(got, per_row, "{rows}: as it errs asked of each row");
        }
    }
    let exists = "EXISTS (SELECT 1 FROM o WHERE o.id = {t}.id AND o.v * 2 > 0)";
    let empty = format!("SELECT id FROM e WHERE {}", exists.replace("{t}", "e"));
    assert_eq!(ids(&db, &empty), Ok(vec![]), "{empty}: no row needs it");
    let rows = format!("SELECT id FROM t WHERE {}", exists.replace("{t}", "t"));
    let got = ids(&db, &rows);
    assert!(
        got.as_ref().is_err_and(|e| e.contains("overflow")),
        "{rows}: {got:?}"
    );
    let per_row = format!(
        "SELECT id FROM t WHERE ({}) OR id < 0",
        exists.replace("{t}", "t")
    );
    assert_eq!(
        got,
        ids(&db, &per_row),
        "{rows}: as it errs asked of each row"
    );
}

static TICKS: AtomicUsize = AtomicUsize::new(0);

/// i64::MAX on its first call in a statement, 1 on its second, 0 after
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
            "test volatile",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
        .non_deterministic()
    }
    fn evaluate(&self, _args: &[Value]) -> stoolap::core::Result<Value> {
        Ok(Value::Integer(
            match TICKS.fetch_add(1, Ordering::Relaxed) {
                0 => i64::MAX,
                1 => 1,
                _ => 0,
            },
        ))
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Tick)
    }
}

#[test]
fn a_block_error_is_raised_again_without_reading_the_block_again() {
    let _turn = COUNTED.lock().unwrap_or_else(|e| e.into_inner());
    // Aggregates resolve functions in the global registry; no other test
    // here calls TICK
    global_registry().register_scalar::<Tick>();
    let registry = FunctionRegistry::new();
    registry.register_scalar::<Tick>();
    let mut engine = MVCCEngine::in_memory();
    stoolap::storage::traits::Engine::open(&mut engine).unwrap();
    let executor = Executor::with_function_registry(Arc::new(engine), Arc::new(registry));
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, i INTEGER)",
        "INSERT INTO t VALUES (1, 5), (2, 6)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO m VALUES (1, 1), (2, 2)",
    ] {
        executor.execute(sql).unwrap();
    }
    for test in ["> ANY", ">= ALL", "IN", "NOT IN", "= ANY", "<> ALL"] {
        let sql = format!("SELECT id FROM t WHERE i {test} (SELECT SUM(TICK(v)) FROM m)");
        TICKS.store(0, Ordering::Relaxed);
        let error = match executor.execute(&sql) {
            Err(e) => Some(e),
            Ok(mut rows) => {
                while rows.next() {}
                rows.last_error()
            }
        };
        assert!(
            error
                .as_ref()
                .is_some_and(|e| e.to_string().contains("overflow")),
            "{sql}: {error:?}"
        );
        assert_eq!(
            TICKS.load(Ordering::Relaxed),
            2,
            "{sql}: the block ran once"
        );
    }
}

#[test]
fn a_long_list_a_cell_searches_stays_the_programs() {
    let _turn = COUNTED.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://prepared_set_scan_float").unwrap();
    for sql in [
        "CREATE TABLE fl (id INTEGER PRIMARY KEY, f FLOAT, i INTEGER)",
        "INSERT INTO fl SELECT g.value, CASE WHEN g.value % 13 = 0 THEN NULL \
         ELSE (g.value % 400) * 0.5 END, g.value % 400 FROM generate_series(1, 3000) g",
        "CREATE TABLE fm (id INTEGER PRIMARY KEY, v FLOAT, n INTEGER)",
        "INSERT INTO fm SELECT g.value, g.value * 0.75, g.value FROM generate_series(1, 200) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // Whether the scan takes each set: a float, or an integer against a
    // fractional member, is compared with every member, so a long set stays
    // the program's; a positive IN is read by its members' rows
    for (condition, scan_takes) in [
        ("f IN (SELECT v FROM fm)", false),
        ("f NOT IN (SELECT v FROM fm)", false),
        ("f IN (SELECT n FROM fm)", false),
        ("f NOT IN (SELECT n FROM fm)", false),
        ("f = ANY (SELECT v FROM fm WHERE id > 20)", false),
        ("f <> ALL (SELECT n FROM fm WHERE id > 20)", false),
        ("f IN (SELECT v FROM fm WHERE id <= 8)", false),
        ("f NOT IN (SELECT v FROM fm WHERE id <= 8)", true),
        ("f <> ALL (SELECT v FROM fm WHERE id <= 8)", true),
        ("i NOT IN (SELECT v FROM fm)", false),
        ("i <> ALL (SELECT v FROM fm WHERE id > 20)", false),
        ("i NOT IN (SELECT n FROM fm)", true),
        ("i NOT IN (SELECT v FROM fm WHERE id <= 8)", true),
        ("i <> ALL (SELECT v * 4 FROM fm)", true),
    ] {
        let sql = format!("SELECT id FROM fl WHERE {condition}");
        let before = prepared_strategy_runs(PreparedStrategy::ScanTakesMembers);
        let got = ids(&db, &sql);
        let taken = prepared_strategy_runs(PreparedStrategy::ScanTakesMembers) > before;
        let per_row = ids(
            &db,
            &format!("SELECT id FROM fl WHERE ({condition}) OR id < 0"),
        );
        assert!(got.as_ref().is_ok_and(|rows| !rows.is_empty()), "{sql}");
        assert_eq!(got, per_row, "{sql}");
        assert_eq!(taken, scan_takes, "{sql}: taken by the scan");
    }
}

#[test]
fn a_set_conjunct_the_scan_takes_reaches_the_scan_strategy() {
    let _turn = COUNTED.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://prepared_set_scan_reached").unwrap();
    fill(&db);
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, Score INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 2), (2, 8)", ())
        .unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM u WHERE Score > ANY (SELECT v FROM m WHERE k = 1)"
        ),
        Ok(vec![2])
    );
    for sql in [
        "SELECT id FROM u WHERE Score > ANY (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i > ANY (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE f <= ALL (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i NOT IN (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i = ANY (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i <> ALL (SELECT v FROM m WHERE k = 1)",
        "SELECT id FROM t WHERE i > ANY (SELECT v FROM m WHERE k = 5)",
    ] {
        let before = prepared_strategy_runs(PreparedStrategy::ScanTakesMembers);
        ids(&db, sql).unwrap();
        assert!(
            prepared_strategy_runs(PreparedStrategy::ScanTakesMembers) > before,
            "{sql}: the scan took the set"
        );
    }
}
