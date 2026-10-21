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

//! A subquery in an UPDATE or DELETE sees the tables as the statement
//! found them, the target included, whatever the statement writes.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db
}

fn rows(db: &Database) -> Vec<String> {
    db.query("SELECT id, b FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            let b: Option<i64> = r.get(1).unwrap();
            format!(
                "{}:{}",
                r.get::<i64>(0).unwrap(),
                b.map_or("NULL".into(), |b| b.to_string())
            )
        })
        .collect()
}

/// Each statement reading its own target, with the rows it leaves when its
/// subqueries see the table as the statement found it
const SELF_READS: &[(&str, &str)] = &[
    (
        "DELETE FROM t WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.id = t.id - 1)",
        "1:10",
    ),
    (
        "UPDATE t SET b = b + 1 WHERE b = (SELECT t2.b FROM t t2 WHERE t2.id = t.id - 1) + 10",
        "1:10 2:21 3:31 4:41",
    ),
    (
        "UPDATE t SET b = (SELECT t2.b FROM t t2 WHERE t2.id = t.id - 1)",
        "1:NULL 2:10 3:20 4:30",
    ),
    (
        "UPDATE t SET b = (SELECT SUM(t2.b) FROM t t2 WHERE t2.id < t.id) WHERE b > (SELECT MIN(b) FROM t)",
        "1:10 2:10 3:30 4:60",
    ),
];

#[test]
fn a_subquery_reads_the_target_as_the_statement_found_it() {
    let mut wrong = Vec::new();
    for (i, (sql, want)) in SELF_READS.iter().enumerate() {
        for explicit in [false, true] {
            let db = setup(&format!("dml_self_read_{i}_{explicit}"));
            if explicit {
                let mut tx = db.begin().unwrap();
                tx.execute(sql, ()).unwrap();
                tx.commit().unwrap();
            } else {
                db.execute(sql, ()).unwrap();
            }
            let got = rows(&db).join(" ");
            if got != *want {
                wrong.push(format!(
                    "{sql} (transaction: {explicit}): {got}, want {want}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_set_may_mix_correlated_and_uncorrelated_subqueries() {
    let db = setup("dml_mixed_set");
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO w VALUES (1, 0, 0), (2, 0, 0), (4, 0, 0)", ())
        .unwrap();
    db.execute(
        "UPDATE w SET a = (SELECT MAX(x) FROM u), b = (SELECT COUNT(*) FROM u WHERE u.id = w.id)",
        (),
    )
    .unwrap();
    let got: Vec<String> = db
        .query("SELECT id, a, b FROM w ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            let (id, a, b): (i64, i64, i64) =
                (r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap());
            format!("{id}:{a}:{b}")
        })
        .collect();
    assert_eq!(got.join(" "), "1:3:1 2:3:1 4:3:0");
}

/// Each with a WHERE conjunct that overflows on the row it reaches, beside
/// the subqueries the prepared statement runs
const OVERFLOWING: &[&str] = &[
    "SELECT id FROM v WHERE ABS(x) >= 0 AND (SELECT u.id FROM u WHERE u.id = v.id) = 1",
    "SELECT id FROM v WHERE ABS(x) >= 0 AND EXISTS (SELECT 1 FROM u WHERE u.id = v.id)",
    "UPDATE v SET b = (SELECT u.id FROM u WHERE u.id = v.id) WHERE ABS(v.x) >= 0 AND (SELECT u.id FROM u WHERE u.id = v.id) = 1",
    "UPDATE v SET b = 5 WHERE ABS(v.x) >= 0 AND (SELECT u.id FROM u WHERE u.id = v.id) = 1",
    "UPDATE v SET b = (SELECT u.id FROM u WHERE u.id = v.id) WHERE ABS(v.x) >= 0",
    "DELETE FROM v WHERE ABS(v.x) >= 0 AND (SELECT u.id FROM u WHERE u.id = v.id) = 1",
    "DELETE FROM v WHERE ABS(v.x) >= 0 AND EXISTS (SELECT 1 FROM u WHERE u.id = v.id)",
];

#[test]
fn a_row_the_transaction_rewrote_to_match_is_written() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        for (sql, want) in [
            (
                "UPDATE v SET b = 9 WHERE x = 1 AND (SELECT u.id FROM u WHERE u.id = v.id) = 1",
                vec![(1, 9), (2, 0)],
            ),
            (
                "DELETE FROM v WHERE x = 1 AND (SELECT u.id FROM u WHERE u.id = v.id) = 1",
                vec![(2, 0)],
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let db = if sealed {
                Database::open(&format!("file://{}", dir.path().display())).unwrap()
            } else {
                Database::open("memory://dml_rewritten_to_match").unwrap()
            };
            for setup in [
                "CREATE TABLE v (id INTEGER PRIMARY KEY, x INTEGER, b INTEGER)",
                "CREATE TABLE u (id INTEGER PRIMARY KEY)",
                "INSERT INTO v VALUES (1, 0, 0), (2, 1, 0)",
                "INSERT INTO u VALUES (1)",
            ] {
                db.execute(setup, ()).unwrap();
            }
            if sealed {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            let mut tx = db.begin().unwrap();
            tx.execute("UPDATE v SET x = 1 WHERE id = 1", ()).unwrap();
            let written = tx.execute(sql, ()).unwrap();
            let left = ids_and_b(tx.query("SELECT id, b FROM v ORDER BY id", ()));
            tx.rollback().unwrap();
            if written != 1 || left != want {
                wrong.push(format!(
                    "{sql} (sealed: {sealed}): wrote {written}, left {left:?}"
                ));
            }
            db.execute("DROP TABLE v", ()).unwrap();
            db.execute("DROP TABLE u", ()).unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn ids_and_b(rows: stoolap::Result<stoolap::Rows>) -> Vec<(i64, i64)> {
    rows.unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap())
        })
        .collect()
}

#[test]
fn a_where_error_fails_a_prepared_statement_and_writes_nothing() {
    let mut wrong = Vec::new();
    for (i, sql) in OVERFLOWING.iter().enumerate() {
        for explicit in [false, true] {
            let db = setup(&format!("dml_where_error_{i}_{explicit}"));
            db.execute(
                "CREATE TABLE v (id INTEGER PRIMARY KEY, x INTEGER, b INTEGER)",
                (),
            )
            .unwrap();
            db.execute(
                "INSERT INTO v VALUES (1, -9223372036854775808, 0), (2, 5, 0)",
                (),
            )
            .unwrap();
            let (failed, left) = if explicit {
                let mut tx = db.begin().unwrap();
                let failed = if sql.starts_with("SELECT") {
                    tx.query(sql, ())
                        .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>())
                        .is_err()
                } else {
                    tx.execute(sql, ()).is_err()
                };
                let left = ids_and_b(tx.query("SELECT id, b FROM v ORDER BY id", ()));
                tx.rollback().unwrap();
                (failed, left)
            } else {
                let failed = if sql.starts_with("SELECT") {
                    db.query(sql, ())
                        .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>())
                        .is_err()
                } else {
                    db.execute(sql, ()).is_err()
                };
                (
                    failed,
                    ids_and_b(db.query("SELECT id, b FROM v ORDER BY id", ())),
                )
            };
            if !failed || left != [(1, 0), (2, 0)] {
                wrong.push(format!(
                    "{sql} (transaction: {explicit}): failed {failed}, left {left:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_scalar_subquery_of_two_rows_fails_the_statement_and_writes_nothing() {
    let db = setup("dml_two_rows");
    for sql in [
        "UPDATE t SET b = (SELECT x FROM u WHERE u.id >= t.id - 1)",
        "DELETE FROM t WHERE b > (SELECT x FROM u WHERE u.id <= t.id)",
    ] {
        assert!(db.execute(sql, ()).is_err(), "{sql} was accepted");
        assert_eq!(rows(&db).join(" "), "1:10 2:20 3:30 4:40", "{sql} wrote");
    }
}

#[test]
fn a_prepared_dml_statement_reads_each_execution_s_parameters() {
    let db = setup("dml_params");
    let update = db
        .prepare("UPDATE t SET b = (SELECT x FROM u WHERE u.id = $1) + t.id WHERE id = $2")
        .unwrap();
    update.execute((1i64, 1i64)).unwrap();
    update.execute((3i64, 2i64)).unwrap();
    let delete = db
        .prepare("DELETE FROM t WHERE b > (SELECT MAX(x) FROM u WHERE u.id < $1)")
        .unwrap();
    delete.execute((3i64,)).unwrap();
    assert_eq!(rows(&db).join(" "), "1:2");
}

#[test]
fn a_cached_dml_statement_follows_a_schema_change() {
    let db = setup("dml_schema_change");
    let sql = "UPDATE t SET b = (SELECT MAX(x) FROM u WHERE u.id <= t.id)";
    db.execute(sql, ()).unwrap();
    db.execute("ALTER TABLE u ADD COLUMN y INTEGER DEFAULT 0", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (4, 9, 0)", ()).unwrap();
    db.execute(sql, ()).unwrap();
    assert_eq!(rows(&db).join(" "), "1:1 2:2 3:3 4:9");
}

static TICKS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
/// The counter is process-wide: the tests that read it take turns
static TICK_TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    fn evaluate(
        &self,
        _args: &[stoolap::core::Value],
    ) -> stoolap::core::Result<stoolap::core::Value> {
        let tick = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        Ok(stoolap::core::Value::Integer(tick))
    }
    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(Tick)
    }
}

#[test]
fn an_uncorrelated_subquery_answers_once_for_the_rows_it_chooses_and_writes() {
    use std::sync::Arc;
    let registry = stoolap::functions::registry::FunctionRegistry::new();
    registry.register_scalar::<Tick>();
    let mut engine = stoolap::storage::mvcc::engine::MVCCEngine::in_memory();
    stoolap::storage::traits::Engine::open(&mut engine).unwrap();
    let executor =
        stoolap::executor::Executor::with_function_registry(Arc::new(engine), Arc::new(registry));
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO t VALUES (1, 0), (2, 0), (3, 0)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY)",
        "INSERT INTO u VALUES (1)",
    ] {
        executor.execute(sql).unwrap();
    }
    let _ticks = TICK_TURN.lock().unwrap_or_else(|e| e.into_inner());
    TICKS.store(0, std::sync::atomic::Ordering::Relaxed);
    executor
        .execute("UPDATE t SET b = 9 WHERE (SELECT TICK() FROM u) = 1")
        .unwrap();
    let mut rows = executor
        .execute("SELECT COUNT(*) FROM t WHERE b = 9")
        .unwrap();
    assert!(rows.next());
    assert_eq!(
        rows.row().get(0).cloned(),
        Some(stoolap::core::Value::Integer(3)),
        "the rows the subquery chose are the rows it let the update write"
    );
}

#[test]
fn a_correlated_volatile_where_answers_once_for_the_row_it_writes() {
    use std::sync::Arc;
    let registry = stoolap::functions::registry::FunctionRegistry::new();
    registry.register_scalar::<Tick>();
    let mut engine = stoolap::storage::mvcc::engine::MVCCEngine::in_memory();
    stoolap::storage::traits::Engine::open(&mut engine).unwrap();
    let executor =
        stoolap::executor::Executor::with_function_registry(Arc::new(engine), Arc::new(registry));
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO t VALUES (1, 0), (2, 0), (3, 0)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY)",
        "INSERT INTO u VALUES (1)",
    ] {
        executor.execute(sql).unwrap();
    }
    let _ticks = TICK_TURN.lock().unwrap_or_else(|e| e.into_inner());
    TICKS.store(0, std::sync::atomic::Ordering::Relaxed);
    executor
        .execute(
            "UPDATE t SET b = (SELECT u.id FROM u WHERE u.id = t.id) \
             WHERE (SELECT TICK() FROM u WHERE u.id = t.id) = 1",
        )
        .unwrap();
    let mut rows = executor.execute("SELECT b FROM t WHERE id = 1").unwrap();
    assert!(rows.next());
    assert_eq!(
        rows.row().get(0).cloned(),
        Some(stoolap::core::Value::Integer(1)),
        "the row its WHERE passed is the row the update wrote"
    );
}
