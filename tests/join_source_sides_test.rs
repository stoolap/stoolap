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

//! A join with a side that is not a table, a CTE, a view, a derived table,
//! a VALUES list or a table function, runs its subqueries as the plan's
//! blocks and answers as the same join over a table holding the side's rows

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{general_path_hits, GeneralPath};
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const MECHANISMS: [GeneralPath; 15] = [
    GeneralPath::CorrelatedScan,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::PushdownConstant,
    GeneralPath::ProjectionEvaluator,
    GeneralPath::OrderByEvaluator,
    GeneralPath::AntiJoin,
    GeneralPath::ExistsSemiJoin,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
    GeneralPath::ScalarSubquery,
    GeneralPath::SelectFold,
    GeneralPath::CorrelatedExpression,
    GeneralPath::LiftedSubqueries,
];

fn hits() -> Vec<usize> {
    MECHANISMS
        .iter()
        .map(|path| general_path_hits(*path))
        .collect()
}

/// The answer of `sql` and the general path mechanisms it ran
fn run(db: &Database, sql: &str) -> (Result<Vec<String>, String>, Vec<GeneralPath>) {
    let before = hits();
    let got = answer(db, sql);
    let ran = MECHANISMS
        .iter()
        .zip(before.iter().zip(hits()))
        .filter(|(_, (before, now))| now > *before)
        .map(|(path, _)| *path)
        .collect();
    (got, ran)
}

/// t joins the side, w is what the subqueries read; the side's rows are
/// u's ids, 1 to 3
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 4, 30), (4, NULL, 40), (5, 3, 50), (6, 1, 60)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
    "INSERT INTO w VALUES (1, 1, 1), (2, 2, 6), (3, 3, 5), (4, 1, 2), (5, 2, 2)",
    "CREATE VIEW uv AS SELECT id FROM u",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://join_source_sides_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// Each side as `{with}` and `{s}` write it, the side named s
const SIDES: &[(&str, &str, &str)] = &[
    ("cte", "WITH s AS (SELECT id FROM u) ", "s"),
    ("cte not inlined", "WITH s (id) AS (SELECT id FROM u) ", "s"),
    (
        "derived over every column",
        "",
        "(SELECT id, x FROM u) AS s",
    ),
    ("view", "", "uv AS s"),
    ("derived", "", "(SELECT id FROM u) AS s"),
    ("values", "", "(VALUES (1), (2), (3)) AS s (id)"),
    ("table function", "", "generate_series(1, 3) AS s (id)"),
];

/// Each statement over t and the side s
const SHAPES: &[&str] = &[
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a WHERE t.b > (SELECT MIN(y) * 10 FROM w)",
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a \
     WHERE EXISTS (SELECT 1 FROM w WHERE w.x = s.id AND w.y = t.id)",
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a WHERE s.id IN (SELECT x FROM w WHERE y > 4)",
    "SELECT t.id, s.id FROM t JOIN {s} ON s.id = t.a \
     WHERE s.id < (SELECT MAX(w.x) FROM w WHERE w.y = s.id)",
    "SELECT t.id, s.id FROM t LEFT JOIN {s} ON s.id = t.a \
     WHERE s.id IS NULL OR s.id IN (SELECT x FROM w WHERE y = 2)",
    "SELECT t.id, s.id FROM t LEFT JOIN {s} ON s.id = t.a AND s.id < (SELECT MAX(x) FROM w)",
    "SELECT t.id, (SELECT COUNT(*) FROM w WHERE w.x = s.id) FROM t JOIN {s} ON s.id = t.a",
    "SELECT s.id, t.id FROM {s} JOIN t ON t.a = s.id WHERE NOT EXISTS (SELECT 1 FROM w WHERE w.y = t.id)",
    "SELECT s.id, t.id FROM {s} LEFT JOIN t ON t.a = s.id AND t.b > 10 \
     WHERE t.id IS NULL OR t.id IN (SELECT y FROM w)",
    "SELECT s.id, t.id FROM {s} JOIN t ON t.a = s.id \
     WHERE s.id + t.id > ALL (SELECT y FROM w WHERE w.x = s.id)",
];

/// The rows of `sql` as sorted text, or the error
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get::<Option<String>>(i) {
                    Ok(value) => value.unwrap_or_else(|| "NULL".into()),
                    Err(e) => format!("<{e}>"),
                })
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    out.sort();
    Ok(out)
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

#[test]
fn a_join_over_a_side_that_is_not_a_table_runs_its_blocks() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir))] {
        for &(kind, with, side) in SIDES {
            for shape in SHAPES {
                let sql = format!("{with}{}", shape.replace("{s}", side));
                let (want, table_ran) = run(&db, &shape.replace("{s}", "u AS s"));
                let (got, ran) = run(&db, &sql);
                if got != want {
                    wrong.push(format!(
                        "{context} {kind}: {sql}: gave {got:?}, a table side {want:?}"
                    ));
                }
                // The blocks' own projection counts as an evaluator, over a table side too
                let evaluator =
                    |path: &&GeneralPath| !matches!(path, GeneralPath::ProjectionEvaluator);
                let table_ran: Vec<_> = table_ran.iter().filter(evaluator).collect();
                if !table_ran.is_empty() {
                    wrong.push(format!(
                        "{context}: {shape} over a table: ran {table_ran:?}"
                    ));
                }
                let ran: Vec<_> = ran.iter().filter(evaluator).collect();
                if !ran.is_empty() {
                    wrong.push(format!("{context} {kind}: {sql}: ran {ran:?}"));
                }
                let line = prepared_line(&db, &sql);
                if line != "Prepared: yes" {
                    wrong.push(format!("{context} {kind}: {sql}: EXPLAIN {line}"));
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
}

#[test]
fn a_join_whose_limit_is_cut_into_a_cte_it_keeps_runs_as_explain_says() {
    let _guard = serial();
    let db = hot("limit_cut");
    let shape = "SELECT c.a, (SELECT MAX(x) FROM w) FROM c LEFT JOIN u ON u.id = c.a \
                 ORDER BY c.a LIMIT 2";
    for (with, line, old) in [
        (
            "WITH c AS (SELECT a, COUNT(*) AS n FROM t GROUP BY a) ",
            "Prepared: yes",
            false,
        ),
        (
            "WITH c (a, n) AS (SELECT a, COUNT(*) FROM t GROUP BY a) ",
            "Prepared: no (a WITH clause)",
            true,
        ),
    ] {
        let sql = format!("{with}{shape}");
        let (got, ran) = run(&db, &sql);
        let evaluator = |path: &&GeneralPath| !matches!(path, GeneralPath::ProjectionEvaluator);
        assert_eq!(got.map(|rows| rows.len()), Ok(2), "{sql}");
        assert_eq!(prepared_line(&db, &sql), line, "{sql}");
        assert_eq!(
            ran.iter().any(|path| evaluator(&path)),
            old,
            "{sql}: ran {ran:?}"
        );
    }
}

/// `sql` as a derived table's statement, which holds no plan, so the
/// general path answers it
fn general(sql: &str) -> String {
    format!("SELECT * FROM ({sql}) AS general")
}

#[test]
fn a_side_without_a_name_runs_on_the_general_path() {
    let _guard = serial();
    let db = hot("unnamed");
    for side in ["(VALUES (1), (2))", "(SELECT id FROM u)"] {
        for shape in [
            "SELECT t.id FROM t JOIN {s} ON t.id = (SELECT MIN(y) FROM w)",
            "SELECT t.id FROM t JOIN {s} ON TRUE WHERE t.id = (SELECT MIN(y) FROM w)",
            "SELECT t.id, (SELECT MAX(y) FROM w) FROM t JOIN {s} ON TRUE",
        ] {
            let sql = shape.replace("{s}", side);
            assert_eq!(answer(&db, &sql), answer(&db, &general(&sql)), "{sql}");
            assert_eq!(
                prepared_line(&db, &sql),
                "Prepared: no (a side without a name)",
                "{sql}"
            );
        }
    }
}

#[test]
fn a_side_read_as_rows_reads_no_more_of_it_than_the_general_path() {
    use stoolap::test_failpoints::{random_draws, tvf_rows};
    let _guard = serial();
    let db = Database::open("memory://join_source_sides_read_once").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 7, g.value FROM generate_series(1, 20) g",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u SELECT g.value, g.value % 1000 FROM generate_series(1, 5000) g",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO w VALUES (1, 1), (2, 2)",
        "CREATE VIEW ru AS SELECT id, x, RANDOM() AS r FROM u",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let count = |sql: &str| answer(&db, sql).map(|rows| rows.len());
    let draws = |sql: &str| {
        let before = random_draws();
        (count(sql), random_draws() - before)
    };
    let mut wrong = Vec::new();
    for (with, side) in [
        ("", "(SELECT id, x, RANDOM() AS r FROM u) AS s"),
        ("", "ru AS s"),
        ("WITH s AS (SELECT id, x, RANDOM() AS r FROM u) ", "s"),
        ("WITH s (id, x, r) AS (SELECT id, x, RANDOM() FROM u) ", "s"),
    ] {
        for shape in [
            "SELECT s.id, (SELECT 1) FROM {s} JOIN t ON t.a = s.id LIMIT 1",
            "SELECT s.id, t.id FROM {s} JOIN t ON t.id = s.x \
             WHERE t.b > (SELECT MIN(x) FROM w) LIMIT 15",
            "SELECT s.id, t.id FROM {s} JOIN t ON t.id = s.x \
             WHERE s.id IN (SELECT x FROM w) LIMIT 1",
            "SELECT s.id FROM {s} JOIN t ON t.a = s.id \
             WHERE s.id > 0 OR s.id = (SELECT MIN(x) FROM w) LIMIT 1",
            "SELECT s.id FROM {s} JOIN t ON t.a = s.id \
             WHERE EXISTS (SELECT 1 FROM w WHERE w.x = s.id) LIMIT 1",
            "SELECT s.id, (SELECT 1) FROM {s} JOIN t ON t.a = s.id WHERE s.id = 3 LIMIT 1",
        ] {
            let sql = format!("{with}{}", shape.replace("{s}", side));
            let reference = format!("{with}{}", general(&shape.replace("{s}", side)));
            let (got, want) = (draws(&sql), draws(&reference));
            if got.0 != want.0 || got.1 > want.1 {
                wrong.push(format!(
                    "{sql}: rows and draws {got:?}, the general path {want:?}"
                ));
            }
        }
    }
    let generated = |sql: &str| {
        let before = tvf_rows();
        (count(sql), tvf_rows() - before)
    };
    for series in [
        "SELECT s.id, (SELECT 1) FROM generate_series(1, 100000) AS s (id) \
         JOIN t ON t.a = s.id LIMIT 1",
        "SELECT s.id, t.id FROM generate_series(1, 100000) AS s (id) JOIN t ON t.a = s.id \
         WHERE s.id IN (SELECT x FROM w) LIMIT 1",
        "SELECT s.id, (SELECT 1) FROM generate_series(1, 1000000) AS s (id) \
         JOIN t ON t.a = s.id WHERE s.id = 1 LIMIT 1",
        "SELECT s.id, (SELECT 1) FROM generate_series(1, 1000000) AS s (id) \
         JOIN t ON t.a = s.id WHERE s.id BETWEEN 2 AND 4 LIMIT 1",
        "SELECT s.id FROM generate_series(1, 100000) AS s (id) JOIN t ON t.a = s.id \
         WHERE s.id > 0 OR s.id = (SELECT MIN(x) FROM w) LIMIT 1",
        "SELECT s.id FROM generate_series(1, 100000) AS s (id) JOIN t ON t.a = s.id \
         WHERE s.id < 10 AND EXISTS (SELECT 1 FROM w WHERE w.x = s.id) LIMIT 1",
    ] {
        let (got, want) = (generated(series), generated(&general(series)));
        if got.0 != want.0 || got.1 > want.1 {
            wrong.push(format!(
                "{series}: rows and generated {got:?}, the general path {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_cte_over_a_join_keeps_its_plan_unless_its_names_wait_for_its_run() {
    let _guard = serial();
    let db = hot("cte_over_join");
    for (with, join, line) in [
        (
            "WITH c AS (SELECT u.id FROM u JOIN w ON w.id = u.id) ",
            "SELECT a.id FROM c a JOIN c b ON a.id = b.id \
             WHERE EXISTS (SELECT 1 FROM t WHERE t.id = a.id)",
            "Prepared: yes",
        ),
        (
            "WITH c AS (SELECT * FROM u JOIN w ON w.id = u.id) ",
            "SELECT t2.id FROM t AS t2 JOIN c ON TRUE \
             WHERE EXISTS (SELECT 1 FROM t WHERE t.id = t2.id)",
            "Prepared: no (a side read with AS OF or not named before it is read)",
        ),
    ] {
        let sql = &format!("{with}{join}");
        let (got, ran) = run(&db, sql);
        assert_eq!(
            got,
            answer(&db, &format!("{with}{}", general(join))),
            "{sql}"
        );
        assert_eq!(prepared_line(&db, sql), line, "{sql}");
        let evaluator = |path: &&GeneralPath| !matches!(path, GeneralPath::ProjectionEvaluator);
        assert_eq!(
            ran.iter().any(|path| evaluator(&path)),
            line != "Prepared: yes",
            "{sql}: the CTE path runs the join without its plan: {ran:?}"
        );
    }
}

mod drops_before_the_join {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use stoolap::core::{Result, Value};
    use stoolap::functions::registry::global_registry;
    use stoolap::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
    };
    use stoolap::Database;

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    /// Returns its argument and counts its calls
    #[derive(Default)]
    struct SideCalls;

    impl ScalarFunction for SideCalls {
        fn name(&self) -> &str {
            "SIDE_CALLS"
        }
        fn info(&self) -> FunctionInfo {
            FunctionInfo::new(
                "SIDE_CALLS",
                FunctionType::Scalar,
                "test counter",
                FunctionSignature::new(
                    FunctionDataType::Integer,
                    vec![FunctionDataType::Any],
                    1,
                    1,
                ),
            )
        }
        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            CALLS.fetch_add(1, Ordering::Relaxed);
            Ok(args[0].clone())
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(SideCalls)
        }
    }

    #[test]
    fn a_side_s_condition_on_values_read_once_drops_its_rows_before_the_join() {
        let _guard = super::serial();
        global_registry().register_scalar::<SideCalls>();
        let db = Database::open("memory://join_source_sides_drops_first").unwrap();
        stoolap::Executor::new(std::sync::Arc::clone(db.engine()))
            .function_registry()
            .register_scalar::<SideCalls>();
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
            "INSERT INTO t SELECT g.value, 1 FROM generate_series(1, 100) g",
            "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
            "INSERT INTO u SELECT g.value, 1 FROM generate_series(1, 500) g",
            "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)",
            "INSERT INTO w VALUES (1, 1), (2, 2)",
            "CREATE VIEW uk AS SELECT id, k FROM u",
        ] {
            db.execute(sql, ()).unwrap();
        }
        let run = |sql: &str| {
            let before = CALLS.load(Ordering::Relaxed);
            let rows = super::answer(&db, sql).map(|rows| rows.len());
            (rows, CALLS.load(Ordering::Relaxed) - before)
        };
        let mut wrong = Vec::new();
        for (with, side) in [
            ("", "(SELECT id, k FROM u) AS s"),
            ("", "uk AS s"),
            ("WITH s (id, k) AS (SELECT id, k FROM u) ", "s"),
        ] {
            for condition in [
                "SIDE_CALLS(s.id) = 1 OR s.id = (SELECT MIN(x) FROM w)",
                "SIDE_CALLS(s.id) = (SELECT MAX(x) FROM w) OR EXISTS (SELECT 1 FROM w WHERE x > 5)",
            ] {
                let shape =
                    format!("SELECT s.id, t.id FROM {side} JOIN t ON t.k = s.k WHERE {condition}");
                let (got, want) = (
                    run(&format!("{with}{shape}")),
                    run(&format!("{with}{}", super::general(&shape))),
                );
                if got.0 != want.0 || got.1 > want.1 {
                    wrong.push(format!(
                        "{with}{shape}: {:?} rows and {} calls, the general path {:?} and {}",
                        got.0, got.1, want.0, want.1
                    ));
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}

mod a_failed_read {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use stoolap::core::{Error, Result, Value};
    use stoolap::functions::registry::global_registry;
    use stoolap::functions::{
        FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
    };
    use stoolap::Database;

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    /// Fails on its first call of a run, then returns its argument
    #[derive(Default)]
    struct FailOnce;

    impl ScalarFunction for FailOnce {
        fn name(&self) -> &str {
            "FAIL_ONCE"
        }
        fn info(&self) -> FunctionInfo {
            FunctionInfo::new(
                "FAIL_ONCE",
                FunctionType::Scalar,
                "test failure",
                FunctionSignature::new(
                    FunctionDataType::Integer,
                    vec![FunctionDataType::Any],
                    1,
                    1,
                ),
            )
        }
        fn evaluate(&self, args: &[Value]) -> Result<Value> {
            match CALLS.fetch_add(1, Ordering::Relaxed) {
                0 => Err(Error::invalid_argument("FAIL_ONCE failed")),
                _ => Ok(args[0].clone()),
            }
        }
        fn clone_box(&self) -> Box<dyn ScalarFunction> {
            Box::new(FailOnce)
        }
    }

    #[test]
    fn an_uncorrelated_block_s_failed_read_is_raised_by_the_row_that_needs_it() {
        let _guard = super::serial();
        global_registry().register_scalar::<FailOnce>();
        let db = Database::open("memory://join_source_sides_failed_read").unwrap();
        stoolap::Executor::new(std::sync::Arc::clone(db.engine()))
            .function_registry()
            .register_scalar::<FailOnce>();
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY)",
            "INSERT INTO t VALUES (1)",
            "CREATE TABLE u (id INTEGER PRIMARY KEY)",
            "INSERT INTO u VALUES (1)",
            "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)",
            "INSERT INTO w VALUES (1, 1)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        let run = |sql: &str| {
            CALLS.store(0, Ordering::Relaxed);
            (super::answer(&db, sql), CALLS.load(Ordering::Relaxed))
        };
        let needed = "SELECT s.id FROM (SELECT id FROM u) AS s JOIN t ON t.id = s.id \
                      WHERE s.id = 0 OR s.id = (SELECT FAIL_ONCE(x) FROM w)";
        let (got, calls) = run(needed);
        assert!(
            got.as_ref().is_err_and(|e| e.contains("FAIL_ONCE failed")),
            "{needed}: the row that needs the block raises its error: {got:?}"
        );
        assert_eq!(calls, 1, "{needed}: the block is read once");
        let unneeded = "SELECT s.id FROM (SELECT id FROM u) AS s JOIN t ON t.id = s.id \
                        WHERE s.id = 1 OR s.id = (SELECT FAIL_ONCE(x) FROM w)";
        assert_eq!(
            run(unneeded).0,
            Ok(vec!["1".to_string()]),
            "{unneeded}: no row needs the block"
        );
    }
}
