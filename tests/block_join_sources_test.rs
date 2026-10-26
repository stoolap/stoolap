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

//! A subquery block whose FROM is a join, or reads a derived table over
//! one, runs as a statement of the prepared blocks and answers as written

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

/// u joins w on id for ids 1 to 3; u 4 has no w, w 5 no u
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 0, 30), (4, 1, 40), (5, 3, NULL)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 1), (2, 2), (3, 3), (4, NULL)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, y INTEGER)",
    "INSERT INTO w VALUES (1, 10), (2, 20), (3, 30), (5, 50)",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://block_join_sources_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir, name: &str) -> Database {
    let path = dir.path().join(name);
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// The rows of `sql` in order, each its values joined by `|`, or the error
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

/// Each statement and its rows in order
const SHAPES: &[(&str, &[&str])] = &[
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id WHERE u.x = t.a) \
         ORDER BY id",
        &["1", "2", "4", "5"],
    ),
    (
        "SELECT id FROM t WHERE (a, b) IN (SELECT u.x, w.y FROM u JOIN w ON w.id = u.id) \
         ORDER BY id",
        &["1", "2"],
    ),
    (
        "SELECT id FROM t WHERE (a, b) IN (SELECT u.x, w.y FROM u JOIN w ON w.id = u.id \
         WHERE u.id <= t.id) ORDER BY id",
        &["1", "2"],
    ),
    (
        "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id \
         WHERE u.x = t.a) ORDER BY id",
        &["3"],
    ),
    (
        "SELECT id FROM t WHERE a IN (SELECT u.x FROM u JOIN w ON w.id = u.id WHERE w.y > 10) \
         ORDER BY id",
        &["2", "5"],
    ),
    (
        "SELECT id, (SELECT MAX(w.y) FROM u JOIN w ON w.id = u.id WHERE u.x = t.a) FROM t \
         ORDER BY id",
        &["1|10", "2|20", "3|NULL", "4|10", "5|30"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id AND w.y = t.b) \
         ORDER BY id",
        &["1", "2", "3"],
    ),
    (
        "SELECT id FROM t WHERE b > (SELECT MAX(w.y) FROM u LEFT JOIN w ON w.id = u.id) \
         ORDER BY id",
        &["4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id \
         JOIN u AS v ON v.id = w.id WHERE v.x = t.a) ORDER BY id",
        &["1", "2", "4", "5"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u JOIN w USING (id) WHERE id = t.a) \
         ORDER BY id",
        &["1", "2", "4", "5"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u NATURAL JOIN w WHERE x = t.a) \
         ORDER BY id",
        &["1", "2", "4", "5"],
    ),
    (
        "SELECT id FROM t WHERE b >= ALL (SELECT w.y FROM u JOIN w ON w.id = u.id \
         WHERE u.x <= t.a) ORDER BY id",
        &["1", "2", "3", "4"],
    ),
    (
        "SELECT id, (SELECT COUNT(*) FROM u JOIN w ON w.id = u.id) FROM t ORDER BY id",
        &["1|3", "2|3", "3|3", "4|3", "5|3"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT u.x, w.y FROM u JOIN w \
         ON w.id = u.id) AS q WHERE q.x = t.a) ORDER BY id",
        &["1", "2", "4", "5"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT id, y FROM u JOIN w USING (id)) \
         AS q WHERE q.id = t.a AND q.y > 10) ORDER BY id",
        &["2", "5"],
    ),
];

#[test]
fn a_block_over_a_join_runs_prepared_and_answers_as_written() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir, "matrix"))] {
        for &(sql, want) in SHAPES {
            let before: Vec<usize> = MECHANISMS.iter().map(|p| general_path_hits(*p)).collect();
            let got = answer(&db, sql);
            let ran: Vec<_> = MECHANISMS
                .iter()
                .zip(before)
                .filter(|(path, before)| general_path_hits(**path) > *before)
                .map(|(path, _)| *path)
                .collect();
            let want: Vec<String> = want.iter().map(|row| row.to_string()).collect();
            if got != Ok(want.clone()) {
                wrong.push(format!("{context}: {sql}: gave {got:?}, want {want:?}"));
            }
            if !ran.is_empty() {
                wrong.push(format!("{context}: {sql}: ran {ran:?}"));
            }
            let line = prepared_line(&db, sql);
            if line != "Prepared: yes" {
                wrong.push(format!("{context}: {sql}: EXPLAIN {line}"));
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
fn a_write_reads_a_block_over_a_join() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for (write, read, want) in [
        (
            "UPDATE t SET b = 0 WHERE EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id \
             WHERE u.x = t.a)",
            "SELECT id FROM t WHERE b = 0 ORDER BY id",
            &["1", "2", "4", "5"][..],
        ),
        (
            "DELETE FROM t WHERE NOT EXISTS (SELECT 1 FROM u JOIN w USING (id) WHERE id = t.a)",
            "SELECT id FROM t ORDER BY id",
            &["1", "2", "4", "5"][..],
        ),
    ] {
        let db = hot(&format!("write_{}", wrong.len() + write.len()));
        let before: Vec<usize> = MECHANISMS.iter().map(|p| general_path_hits(*p)).collect();
        if let Err(e) = db.execute(write, ()) {
            wrong.push(format!("{write}: {e}"));
            continue;
        }
        let ran: Vec<_> = MECHANISMS
            .iter()
            .zip(before)
            .filter(|(path, before)| general_path_hits(**path) > *before)
            .map(|(path, _)| *path)
            .collect();
        let want: Vec<String> = want.iter().map(|row| row.to_string()).collect();
        let got = answer(&db, read);
        if got != Ok(want.clone()) {
            wrong.push(format!("{write}: left {got:?}, want {want:?}"));
        }
        if !ran.is_empty() {
            wrong.push(format!("{write}: ran {ran:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_uncorrelated_block_over_a_join_runs_once_per_run() {
    use stoolap::test_failpoints::random_draws;
    let _guard = serial();
    let db = hot("once");
    let mut wrong = Vec::new();
    for (sql, rows) in [
        (
            "SELECT id FROM t WHERE a IN (SELECT u.x FROM u JOIN w ON w.id = u.id \
             WHERE RANDOM() >= 0) ORDER BY id",
            4,
        ),
        (
            "SELECT id, (SELECT COUNT(*) FROM u JOIN w ON w.id = u.id WHERE RANDOM() >= 0) FROM t \
             ORDER BY id",
            5,
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id \
             WHERE RANDOM() < 0) ORDER BY id",
            0,
        ),
    ] {
        let before = random_draws();
        let got = answer(&db, sql).map(|found| found.len());
        let draws = random_draws() - before;
        if got != Ok(rows) || draws != 3 {
            wrong.push(format!(
                "{sql}: {got:?} rows and {draws} draws, want {rows} rows and the 3 of one run"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_kept_answer_of_a_block_over_a_join_follows_its_parameters() {
    let _guard = serial();
    let db = hot("params");
    let read = |sql: &str, bound: i64| -> Vec<String> {
        let rows = db.query(sql, (bound,)).unwrap();
        rows.map(|row| row.unwrap().get::<i64>(0).unwrap().to_string())
            .collect()
    };
    let mut wrong = Vec::new();
    for (sql, (low, first), (high, then)) in [
        (
            "SELECT id FROM t WHERE a IN (SELECT u.x FROM u JOIN w ON w.id = u.id \
             WHERE w.y <= $1) ORDER BY id",
            (10, vec!["1", "4"]),
            (30, vec!["1", "2", "4", "5"]),
        ),
        (
            "SELECT id FROM t WHERE b > (SELECT MAX(w.y) FROM u JOIN w ON w.id = u.id \
             WHERE w.y <= $1) ORDER BY id",
            (10, vec!["2", "3", "4"]),
            (30, vec!["4"]),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id \
             WHERE w.y > $1)",
            (10, vec!["5"]),
            (30, vec!["0"]),
        ),
        (
            "SELECT id FROM t WHERE a IN (SELECT q.x FROM (SELECT u.x FROM u JOIN w \
             ON w.id = u.id WHERE w.y <= $1) AS q) ORDER BY id",
            (10, vec!["1", "4"]),
            (30, vec!["1", "2", "4", "5"]),
        ),
        (
            "SELECT id FROM t WHERE b IN (SELECT q.s FROM u JOIN (SELECT w.id, SUM(w.y) \
             OVER win AS s FROM w WINDOW win AS (ORDER BY w.id ROWS BETWEEN $1 PRECEDING \
             AND CURRENT ROW)) AS q ON q.id = u.id) ORDER BY id",
            (0, vec!["1", "2", "3"]),
            (1, vec!["1", "3"]),
        ),
        (
            "SELECT t.id FROM t JOIN w AS z ON z.id = t.id AND t.b IN (SELECT q.s FROM u \
             JOIN (SELECT w.id, SUM(w.y) OVER win AS s FROM w WINDOW win AS (ORDER BY w.id \
             ROWS BETWEEN $1 PRECEDING AND CURRENT ROW)) AS q ON q.id = u.id) ORDER BY t.id",
            (0, vec!["1", "2", "3"]),
            (1, vec!["1", "3"]),
        ),
    ] {
        for (bound, want) in [(low, &first), (high, &then)] {
            let got = read(sql, bound);
            if got != *want {
                wrong.push(format!(
                    "{sql} with $1 = {bound}: gave {got:?}, want {want:?}"
                ));
            }
        }
    }
    let outer = "SELECT id FROM t WHERE b IN (SELECT q.s FROM u JOIN (SELECT w.id, SUM(w.y) \
                 OVER win AS s FROM w WINDOW win AS (ORDER BY w.id ROWS BETWEEN t.a PRECEDING \
                 AND CURRENT ROW)) AS q ON q.id = u.id) ORDER BY id";
    let got = answer(&db, outer);
    if got != Ok(vec!["1".to_string(), "3".to_string()]) {
        wrong.push(format!("{outer}: gave {got:?}, want [1, 3]"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_in_over_a_join_narrows_the_statement_as_one_over_a_table_does() {
    use stoolap::test_failpoints::{
        prepared_rows_narrowed, prepared_strategy_runs, PreparedStrategy,
    };
    let _guard = serial();
    let db = hot("narrows");
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE id IN (SELECT u.id FROM u JOIN w ON w.id = u.id) ORDER BY id",
            vec!["1", "2", "3"],
        ),
        (
            "SELECT id FROM t WHERE id NOT IN (SELECT u.id FROM u JOIN w ON w.id = u.id) \
             ORDER BY id",
            vec!["4", "5"],
        ),
        (
            "SELECT id FROM t WHERE a IN (SELECT u.x FROM u JOIN w ON w.id = u.id) ORDER BY id",
            vec!["1", "2", "4", "5"],
        ),
    ] {
        let before = prepared_rows_narrowed();
        let got = answer(&db, sql);
        if got != Ok(want.iter().map(|s| s.to_string()).collect()) {
            wrong.push(format!("{sql}: gave {got:?}, want {want:?}"));
        }
        if prepared_rows_narrowed() == before {
            wrong.push(format!("{sql}: read t whole, not by the members' keys"));
        }
    }
    let scanned = "SELECT id FROM t WHERE a NOT IN (SELECT u.x FROM u JOIN w ON w.id = u.id) \
                   ORDER BY id";
    let before = prepared_strategy_runs(PreparedStrategy::ScanTakesMembers);
    let got = answer(&db, scanned);
    if got != Ok(vec!["3".into()]) {
        wrong.push(format!("{scanned}: gave {got:?}"));
    }
    if prepared_strategy_runs(PreparedStrategy::ScanTakesMembers) == before {
        wrong.push(format!("{scanned}: the scan did not take the members"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn an_exists_over_a_join_checks_its_dropped_clauses_as_its_run_names_them() {
    let _guard = serial();
    let db = hot("dropped");
    let refused = |position: &str| Err(format!("invalid argument: {}", position));
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT * FROM u JOIN w USING (id) ORDER BY 3)",
            Ok(vec!["5".to_string()]),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT * FROM u JOIN w USING (id) ORDER BY 4)",
            refused("ORDER BY position 4 is not in the select list"),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT u.* FROM u JOIN w ON w.id = u.id \
             ORDER BY 3)",
            refused("ORDER BY position 3 is not in the select list"),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT v.x FROM u JOIN u AS v ON v.id = u.id \
             GROUP BY v.x)",
            Ok(vec!["5".to_string()]),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT u.x FROM u JOIN u AS v ON v.id = u.id \
             GROUP BY v.x)",
            Err("Compile error: Column 'u.x' not found".to_string()),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT v.x FROM u CROSS JOIN (VALUES (1)) \
             AS v(x) GROUP BY x)",
            Ok(vec!["5".to_string()]),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT u.x FROM u CROSS JOIN (VALUES (1)) \
             AS v(x) GROUP BY x)",
            Err("Compile error: Column 'u.x' not found".to_string()),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT g.value FROM u CROSS JOIN \
             generate_series(1, 2) AS g GROUP BY value)",
            Ok(vec!["5".to_string()]),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT q.x FROM u CROSS JOIN \
             (SELECT 1 AS x) AS q GROUP BY x)",
            Err("Compile error: Column 'q.x' not found".to_string()),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT u.x AS y FROM u CROSS JOIN \
             (VALUES (5)) AS v(y) WHERE y = 1)",
            Ok(vec!["0".to_string()]),
        ),
        (
            "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT u.x AS z FROM u CROSS JOIN \
             (VALUES (5)) AS v(y) WHERE z = 1)",
            Ok(vec!["5".to_string()]),
        ),
    ] {
        let got = answer(&db, sql);
        if got != want {
            wrong.push(format!("{sql}: gave {got:?}, want {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_block_over_a_join_is_bound_when_prepared_and_run_when_asked() {
    use stoolap::test_failpoints::{prepared_strategy_runs, PreparedStrategy};
    let _guard = serial();
    let db = hot("bound");
    for sql in [
        "CREATE TABLE e (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE one (id INTEGER PRIMARY KEY)",
        "INSERT INTO one VALUES (1)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO m VALUES (1, -9223372036854775808)",
        "CREATE TABLE z (id INTEGER PRIMARY KEY)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let ungrouped = Err("Compile error: Column 'u.x' not found".to_string());
    let overflow = Err("type error: Integer overflow in ABS: -9223372036854775808".to_string());
    let none = Ok(Vec::new());
    let mut wrong = Vec::new();
    // Each shape over a join answers as its twin over one table does
    for (sql, want) in [
        (
            "SELECT id FROM e WHERE EXISTS (SELECT u.x FROM u JOIN u AS v ON v.id = u.id \
             GROUP BY v.x)",
            ungrouped.clone(),
        ),
        (
            "SELECT id FROM e WHERE a IN (SELECT u.x FROM u JOIN u AS v ON v.id = u.id \
             GROUP BY v.x)",
            ungrouped.clone(),
        ),
        (
            "SELECT id, (SELECT u.x FROM u JOIN u AS v ON v.id = u.id GROUP BY v.x) FROM e",
            ungrouped.clone(),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT u.x FROM u JOIN u AS v ON v.id = u.id \
             WHERE u.id = e.id GROUP BY v.x)",
            ungrouped.clone(),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT u.x FROM u WHERE u.id = e.id GROUP BY u.id)",
            Err("Compile error: Column 'x' not found".to_string()),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT u.x FROM u JOIN u AS v ON v.id = u.id \
             WHERE EXISTS (SELECT 1 FROM z) GROUP BY v.x)",
            ungrouped.clone(),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT u.x FROM u WHERE EXISTS (SELECT 1 FROM z) \
             GROUP BY u.id)",
            Err("Compile error: Column 'x' not found".to_string()),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT u.x FROM u JOIN u AS v ON v.id = u.id \
             AND EXISTS (SELECT 1 FROM z) GROUP BY v.x)",
            ungrouped.clone(),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT 1 FROM u JOIN u AS v ON v.id = u.id \
             WHERE u.no_such_column = 1 AND EXISTS (SELECT 1 FROM z))",
            Err("column 'u.no_such_column' not found".to_string()),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT 1 FROM u WHERE u.no_such_column = 1 \
             AND EXISTS (SELECT 1 FROM z))",
            Err("column 'u.no_such_column' not found".to_string()),
        ),
        (
            "SELECT id, (SELECT LENGTH(STRING_AGG(CAST(u.id AS TEXT), CASE WHEN one.id IS NULL \
             THEN CAST(ABS(-9223372036854775807 - 1) AS TEXT) ELSE ',' END)) + ROW_NUMBER() \
             OVER () FROM u JOIN w ON w.id = u.id) FROM one",
            Ok(vec!["1|6".to_string()]),
        ),
        (
            "SELECT id, (SELECT LENGTH(STRING_AGG(CAST(u.id AS TEXT), CASE WHEN one.id IS NULL \
             THEN CAST(ABS(-9223372036854775807 - 1) AS TEXT) ELSE ',' END)) FROM u JOIN w \
             ON w.id = u.id) FROM one",
            Ok(vec!["1|5".to_string()]),
        ),
        (
            "SELECT id, (SELECT ABS(COUNT(*) - 9223372036854775807 - 1) FROM u JOIN w \
             ON w.id = u.id) FROM one",
            Ok(vec!["1|9223372036854775805".to_string()]),
        ),
        (
            "SELECT id, (SELECT ABS(COUNT(*) - 9223372036854775807 - 1) FROM u) FROM one",
            Ok(vec!["1|9223372036854775804".to_string()]),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT v.x FROM u JOIN u AS v ON v.id = u.id \
             GROUP BY v.x)",
            none.clone(),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT 1 FROM m JOIN u ON u.id = m.id \
             WHERE ABS(m.x) > 0)",
            none.clone(),
        ),
        (
            "SELECT id FROM e WHERE EXISTS (SELECT 1 FROM m WHERE ABS(m.x) > 0)",
            none.clone(),
        ),
        (
            "SELECT id FROM e WHERE a IN (SELECT ABS(m.x) FROM m JOIN u ON u.id = m.id)",
            none.clone(),
        ),
        (
            "SELECT id FROM e WHERE a IN (SELECT ABS(m.x) FROM m)",
            none.clone(),
        ),
        (
            "SELECT id FROM one WHERE CASE WHEN id = 0 THEN (SELECT ABS(m.x) FROM m \
             JOIN u ON u.id = m.id LIMIT 1) > 0 ELSE TRUE END",
            Ok(vec!["1".to_string()]),
        ),
        (
            "SELECT id FROM one WHERE CASE WHEN id = 0 THEN (SELECT ABS(m.x) FROM m \
             LIMIT 1) > 0 ELSE TRUE END",
            Ok(vec!["1".to_string()]),
        ),
        (
            "SELECT id FROM one WHERE CASE WHEN id = 0 THEN EXISTS (SELECT 1 FROM m \
             JOIN u ON u.id = m.id WHERE ABS(m.x) > 0) ELSE TRUE END",
            Ok(vec!["1".to_string()]),
        ),
        (
            "SELECT id FROM one WHERE CASE WHEN id = 0 THEN 1 IN (SELECT ABS(m.x) FROM m \
             JOIN u ON u.id = m.id) ELSE TRUE END",
            Ok(vec!["1".to_string()]),
        ),
        (
            "SELECT id FROM one WHERE CASE WHEN id = 0 THEN (SELECT ABS(m.x) + \
             ROW_NUMBER() OVER () FROM m LIMIT 1) > 0 ELSE TRUE END",
            Ok(vec!["1".to_string()]),
        ),
        (
            "SELECT id, CASE WHEN id = 0 THEN (SELECT ABS(m.x) FROM m JOIN u \
             ON u.id = m.id LIMIT 1) END FROM one",
            overflow.clone(),
        ),
        (
            "SELECT id, CASE WHEN id = 0 THEN (SELECT ABS(m.x) FROM m LIMIT 1) END FROM one",
            overflow.clone(),
        ),
        (
            "SELECT CASE WHEN id = 0 THEN (SELECT ABS(m.x) + ROW_NUMBER() OVER () FROM m \
             JOIN u ON u.id = m.id LIMIT 1) ELSE 1 END FROM one",
            Ok(vec!["1".to_string()]),
        ),
        (
            "SELECT CASE WHEN id = 0 THEN (SELECT ABS(m.x) + ROW_NUMBER() OVER () FROM m \
             LIMIT 1) ELSE 1 END FROM one",
            Ok(vec!["1".to_string()]),
        ),
        (
            "SELECT id, (SELECT u.x FROM u JOIN w ON w.id = u.id) FROM e",
            Err("scalar subquery returned more than one row".to_string()),
        ),
        (
            "SELECT id, (SELECT u.x FROM u) FROM e",
            Err("scalar subquery returned more than one row".to_string()),
        ),
        // Over a join an EXISTS's statement computes its SELECT list; the docs say so
        (
            "SELECT id FROM one WHERE EXISTS (SELECT ABS(m.x) FROM m JOIN u ON u.id = m.id)",
            overflow.clone(),
        ),
        (
            "SELECT id FROM one WHERE EXISTS (SELECT ABS(m.x) FROM m)",
            Ok(vec!["1".to_string()]),
        ),
    ] {
        // A second run takes the plan the first one made
        for run in ["first", "second"] {
            let got = answer(&db, sql);
            if got != want {
                wrong.push(format!("{sql} ({run} run): gave {got:?}, want {want:?}"));
            }
        }
    }
    // A LIMIT is checked by the run that reads the block, whatever ran before
    for sql in [
        "SELECT id FROM e WHERE EXISTS (SELECT 1 FROM u JOIN w ON w.id = u.id LIMIT $1)",
        "SELECT id FROM e WHERE EXISTS (SELECT 1 FROM u LIMIT $1)",
        "SELECT id FROM e WHERE a IN (SELECT u.x FROM u JOIN w ON w.id = u.id LIMIT $1)",
        "SELECT id FROM e WHERE a IN (SELECT u.x FROM u LIMIT $1)",
    ] {
        for bound in [-1i64, 1, -1] {
            let got = db
                .query(sql, (bound,))
                .map(|rows| rows.count())
                .map_err(|e| e.to_string());
            if got != Ok(0) {
                wrong.push(format!(
                    "{sql} with $1 = {bound}: gave {got:?}, want no row"
                ));
            }
        }
    }
    // A plan that finds the block's answer kept by another runs no join for it
    let block = "a IN (SELECT u.x FROM u JOIN w ON w.id = u.id WHERE w.y > 10)";
    answer(&db, &format!("SELECT id FROM t WHERE id >= 1 AND {block}")).unwrap();
    let before = prepared_strategy_runs(PreparedStrategy::BlockStatement);
    let got = answer(
        &db,
        &format!("SELECT id FROM t WHERE id >= 2 AND {block} ORDER BY id"),
    );
    if got != Ok(vec!["2".to_string(), "5".to_string()]) {
        wrong.push(format!("the second plan gave {got:?}"));
    }
    if prepared_strategy_runs(PreparedStrategy::BlockStatement) != before {
        wrong.push("the second plan ran the block's join again".to_string());
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_star_over_a_join_is_named_only_by_its_run() {
    let _guard = serial();
    let db = hot("star");
    let thrice = |ids: &[&str]| -> Vec<String> {
        ids.iter()
            .flat_map(|id| std::iter::repeat_n(id.to_string(), 3))
            .collect()
    };
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT t.id FROM t JOIN (SELECT * FROM u JOIN w ON w.id = u.id) AS q ON TRUE \
             WHERE EXISTS (SELECT 1 FROM w WHERE w.id = t.id) ORDER BY t.id",
            thrice(&["1", "2", "3", "5"]),
        ),
        (
            "WITH c AS (SELECT * FROM u JOIN w ON w.id = u.id) SELECT t.id FROM t JOIN c ON TRUE \
             WHERE EXISTS (SELECT 1 FROM w WHERE w.id = t.id) ORDER BY t.id",
            thrice(&["1", "2", "3", "5"]),
        ),
        (
            "WITH c AS (SELECT * FROM u JOIN w ON w.id = u.id) SELECT t.id FROM t JOIN c ON TRUE \
             WHERE EXISTS (SELECT 1 FROM c AS d WHERE t.id = 1) ORDER BY t.id",
            thrice(&["1"]),
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM (SELECT * FROM u JOIN w ON w.id = u.id) \
             AS q WHERE t.a > 1) ORDER BY id",
            vec!["2".to_string(), "5".to_string()],
        ),
    ] {
        let got = answer(&db, sql);
        if got != Ok(want.clone()) {
            wrong.push(format!("{sql}: gave {got:?}, want {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_kept_answer_of_a_block_over_a_join_follows_its_tables() {
    let _guard = serial();
    let db = hot("kept");
    let in_sql = "SELECT id FROM t WHERE a IN (SELECT u.x FROM u JOIN w ON w.id = u.id \
                  WHERE w.y > 10) ORDER BY id";
    let max_sql = "SELECT id FROM t WHERE b > (SELECT MAX(w.y) FROM u JOIN w ON w.id = u.id) \
                   ORDER BY id";
    let exists_sql = "SELECT COUNT(*) FROM t WHERE EXISTS (SELECT 1 FROM u JOIN w \
                      ON w.id = u.id WHERE w.y > 40)";
    let read = |sql: &str| answer(&db, sql).unwrap();
    assert_eq!(read(in_sql), ["2", "5"]);
    assert_eq!(read(max_sql), ["4"]);
    assert_eq!(read(exists_sql), ["0"]);
    // u 4 meets a w row now, with x NULL; w 1 rises above every b
    db.execute("INSERT INTO w VALUES (4, 45)", ()).unwrap();
    db.execute("UPDATE u SET x = 1 WHERE id = 4", ()).unwrap();
    assert_eq!(read(in_sql), ["1", "2", "4", "5"], "{in_sql}");
    assert_eq!(read(max_sql), Vec::<String>::new(), "{max_sql}");
    assert_eq!(read(exists_sql), ["5"], "{exists_sql}");
    // A transaction reads its own writes, and the answers after it reads none
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM w WHERE id = 4", ()).unwrap();
    assert_eq!(read(exists_sql), ["0"], "{exists_sql} in the transaction");
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(read(exists_sql), ["5"], "{exists_sql} after the rollback");
}
