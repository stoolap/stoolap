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

//! A join of more than two sources, a join whose side is itself a join, runs
//! its subqueries as the plan's blocks and answers as the general path does

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

/// t, u and the side s join; w is what the subqueries read; p is a fourth
/// source. NULLs and unmatched keys on every side
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 4, 30), (4, NULL, 40), (5, 3, 50), (6, 1, 60)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9), (5, NULL)",
    "CREATE TABLE r (id INTEGER PRIMARY KEY)",
    "INSERT INTO r VALUES (1), (2), (3)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
    "INSERT INTO w VALUES (1, 1, 1), (2, 2, 6), (3, 3, 5), (4, 1, 2), (5, 2, 2)",
    "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER)",
    "INSERT INTO p VALUES (1, 1), (2, 1), (3, 2), (6, 3)",
    "CREATE VIEW rv AS SELECT id FROM r",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://join_nested_sides_{name}")).unwrap();
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

/// Each kind of source as `{with}` and `{s}` write it, named s, its rows
/// r's ids, 1 to 3
const SIDES: &[(&str, &str, &str)] = &[
    ("table", "", "r AS s"),
    ("cte", "WITH s AS (SELECT id FROM r) ", "s"),
    ("cte not inlined", "WITH s (id) AS (SELECT id FROM r) ", "s"),
    ("view", "", "rv AS s"),
    ("derived", "", "(SELECT id FROM r) AS s"),
    ("values", "", "(VALUES (1), (2), (3)) AS s (id)"),
    ("table function", "", "generate_series(1, 3) AS s (id)"),
];

/// Each statement over three sources: `{f}` is the FROM, `{j1}` and `{j2}`
/// the two joins' kinds
const SHAPES: &[&str] = &[
    "SELECT t.id, u.id, s.id FROM {f} WHERE t.b > (SELECT MIN(y) * 10 FROM w)",
    "SELECT t.id, u.id, s.id FROM {f} WHERE s.id IN (SELECT x FROM w WHERE y > 4)",
    "SELECT t.id, u.id, s.id FROM {f} WHERE EXISTS (SELECT 1 FROM w WHERE w.x = s.id AND w.y = t.id)",
    "SELECT t.id, u.id, s.id FROM {f} WHERE u.x < (SELECT MAX(w.x) + 7 FROM w WHERE w.y = s.id)",
    "SELECT t.id, u.id, s.id FROM {f} WHERE u.id IS NULL OR u.id IN (SELECT x FROM w WHERE y = 2)",
    "SELECT t.id, u.id, s.id FROM {f} WHERE NOT EXISTS (SELECT 1 FROM w WHERE w.y = t.id)",
    "SELECT t.id, u.id, s.id FROM {f} WHERE s.id + t.id > ALL (SELECT y FROM w WHERE w.x = s.id)",
    "SELECT t.id, (SELECT COUNT(*) FROM w WHERE w.x = s.id), u.x FROM {f}",
    "SELECT t.id AS ti, u.id AS ui, s.id AS si, (SELECT COUNT(*) FROM w WHERE w.x = u.id) AS k \
     FROM {f} ORDER BY (SELECT COUNT(*) FROM w WHERE w.x = u.id), ti, ui, si",
];

/// The order an ordered shape's rows are compared in, as its general path
/// reference sorts them by the columns it selects
const ORDERED: &str = " ORDER BY k, ti, ui, si";

/// The FROMs of three sources, the side third or second, with its ON
/// holding a subquery or not
const FROMS: &[&str] = &[
    "t {j1} JOIN u ON u.id = t.a {j2} JOIN {s} ON s.id = u.id",
    "t {j1} JOIN u ON u.id = t.a {j2} JOIN {s} ON s.id = t.a AND s.id < (SELECT MAX(x) FROM w)",
    "t {j1} JOIN u ON u.id = t.a AND u.x > (SELECT MIN(y) FROM w) {j2} JOIN {s} ON s.id = u.id",
    "t {j1} JOIN {s} ON s.id = t.a {j2} JOIN u ON u.id = s.id",
];

const FIRST: &[&str] = &["", "LEFT"];
const SECOND: &[&str] = &["", "LEFT", "RIGHT", "FULL"];

/// The rows of `sql` as text, sorted unless the statement orders them, or
/// the error
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
    if !sql.contains(" ORDER BY ") {
        out.sort();
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

/// `sql` as a derived table's statement, which holds no plan, so the
/// general path answers it, in the order an ordered shape compares
fn general(with: &str, sql: &str) -> String {
    let order = match sql.contains(" ORDER BY ") {
        true => ORDERED,
        false => "",
    };
    format!("{with}SELECT * FROM ({sql}) AS general{order}")
}

/// Every statement of `sqls`, `{with}` before it, answers as the general
/// path; prepared, as EXPLAIN says `line`, it runs no general path mechanism
fn check(
    db: &Database,
    context: &str,
    (with, line): (&str, &str),
    sqls: &[String],
    wrong: &mut Vec<String>,
) {
    for sql in sqls {
        let (want, _) = run(db, &general(with, sql));
        let full = format!("{with}{sql}");
        let (got, ran) = run(db, &full);
        if got != want {
            wrong.push(format!(
                "{context}: {full}: gave {got:?}, the general path {want:?}"
            ));
        }
        if line == "Prepared: yes" && !ran.is_empty() {
            wrong.push(format!("{context}: {full}: ran {ran:?}"));
        }
        let explained = prepared_line(db, &full);
        if explained != line {
            wrong.push(format!(
                "{context}: {full}: EXPLAIN {explained}, want {line}"
            ));
        }
    }
}

#[test]
fn a_join_of_three_sources_runs_its_blocks_and_answers_as_the_general_path() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    let mut count = 0;
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir))] {
        for &(kind, with, side) in SIDES {
            for from in FROMS {
                let mut sqls = Vec::new();
                for j1 in FIRST {
                    for j2 in SECOND {
                        let from = from
                            .replace("{j1}", j1)
                            .replace("{j2}", j2)
                            .replace("{s}", side);
                        for shape in SHAPES {
                            sqls.push(shape.replace("{f}", &from));
                        }
                    }
                }
                count += sqls.len();
                let context = format!("{context} {kind}");
                check(&db, &context, (with, "Prepared: yes"), &sqls, &mut wrong);
            }
        }
    }
    assert!(count > 1000, "{count} statements");
    assert!(
        wrong.is_empty(),
        "{} of {count} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_join_of_four_sources_and_a_cross_join_run_their_blocks() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    let sqls: Vec<String> = [
        "SELECT t.id, p.id FROM t JOIN u ON u.id = t.a JOIN r ON r.id = u.id JOIN p ON p.k = r.id \
         WHERE p.id IN (SELECT y FROM w)",
        "SELECT t.id, p.id FROM t LEFT JOIN u ON u.id = t.a JOIN r ON r.id = t.a \
         LEFT JOIN p ON p.k = r.id AND p.id > (SELECT MIN(y) FROM w) WHERE t.b > 10",
        "SELECT t.id, (SELECT MAX(y) FROM w WHERE w.x = p.k) FROM t JOIN u ON u.id = t.a \
         JOIN r ON r.id = u.id JOIN p ON p.k = r.id",
        "SELECT t.id, u.id, r.id FROM t CROSS JOIN u JOIN r ON r.id = u.id \
         WHERE t.a = u.id AND t.b > (SELECT MIN(y) * 10 FROM w)",
        "SELECT t.id, u.id, r.id FROM t JOIN u ON u.id = t.a CROSS JOIN r \
         WHERE r.id IN (SELECT x FROM w WHERE y > 4)",
        "SELECT t.id, u.id, r.id FROM t JOIN u ON u.id = t.a \
         JOIN r ON r.id = u.id AND r.id IN (SELECT x FROM w WHERE y < 6)",
    ]
    .iter()
    .map(|sql| sql.to_string())
    .collect();
    for (context, db) in [("hot", hot("deep")), ("sealed", sealed(&dir))] {
        check(&db, context, ("", "Prepared: yes"), &sqls, &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_subquery_in_the_output_over_a_joined_side_that_merges_stays_on_the_general_path() {
    let _guard = serial();
    let db = hot("refused");
    let mut wrong = Vec::new();
    for (sql, line) in [
        (
            "SELECT t.id, u.id, (SELECT MIN(y) * 10 FROM w) FROM t JOIN u USING (id) \
             JOIN r ON r.id = t.a",
            "Prepared: no (a NATURAL or USING join whose output holds a subquery)",
        ),
        (
            "SELECT t.id, r.id, (SELECT MAX(y) FROM w) FROM t NATURAL JOIN r \
             JOIN u ON u.id = t.a",
            "Prepared: no (a NATURAL or USING join whose output holds a subquery)",
        ),
    ] {
        check(&db, "hot", ("", line), &[sql.to_string()], &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_subquery_in_a_branch_no_joined_row_takes_is_never_read() {
    use stoolap::test_failpoints::random_draws;
    let _guard = serial();
    let db = hot("branch");
    let mut wrong = Vec::new();
    for sql in [
        "SELECT t.id FROM t LEFT JOIN u ON u.id = t.a \
         WHERE CASE WHEN t.id + COALESCE(u.id, 0) > 0 THEN TRUE ELSE (SELECT RANDOM()) > 2 END",
        "SELECT t.id FROM t LEFT JOIN u ON u.id = t.a JOIN r ON r.id = t.a \
         WHERE CASE WHEN t.id + COALESCE(u.id, 0) > 0 THEN TRUE ELSE (SELECT RANDOM()) > 2 END",
        "SELECT t.id FROM t JOIN u ON u.id = t.a JOIN r ON r.id = u.id \
         WHERE t.id > 0 OR u.x > (SELECT RANDOM() * 100)",
        "SELECT t.id FROM t JOIN u ON u.id = t.a JOIN r ON r.id = u.id \
         WHERE CASE WHEN t.id > 0 THEN TRUE ELSE u.x IN (SELECT RANDOM() FROM w) END",
        "SELECT t.id FROM t JOIN u ON u.id = t.a JOIN r ON r.id = u.id \
         WHERE t.id > 100 AND u.x + r.id IN (SELECT RANDOM() FROM w)",
    ] {
        let before = random_draws();
        let rows = answer(&db, sql).map(|rows| rows.len());
        let drawn = random_draws() - before;
        if drawn != 0 || rows.is_err() {
            wrong.push(format!("{sql}: {rows:?} rows, {drawn} draws, want none"));
        }
        if prepared_line(&db, sql) != "Prepared: yes" {
            wrong.push(format!("{sql}: not prepared"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_joined_side_whose_columns_change_after_preparing_is_refused_or_read_anew() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use stoolap::test_failpoints::{after_join_index_chosen, reset_all};
    let _guard = serial();
    let db = Database::open("memory://join_nested_sides_renamed").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 1)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
        "INSERT INTO u VALUES (1, 10, 20)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "INSERT INTO r VALUES (1)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO w VALUES (1, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT t.id FROM t JOIN u ON u.id = t.a JOIN r ON r.id = t.a \
               WHERE EXISTS (SELECT 1 FROM w WHERE w.k = u.x) LIMIT 10";
    assert_eq!(
        answer(&db, sql),
        Ok(Vec::new()),
        "u.x is 10 before the renames"
    );
    let (other, reader) = (db.clone(), db.clone());
    let fired = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&fired);
    let got = std::thread::spawn(move || {
        after_join_index_chosen(move || {
            for rename in [
                "ALTER TABLE u RENAME COLUMN x TO z",
                "ALTER TABLE u RENAME COLUMN y TO x",
                "ALTER TABLE u RENAME COLUMN z TO y",
            ] {
                other.execute(rename, ()).unwrap();
            }
            seen.store(true, Ordering::SeqCst);
        });
        let got = reader
            .query(sql, ())
            .map(|rows| {
                rows.map(|row| row.unwrap().get::<i64>(0).unwrap())
                    .collect::<Vec<_>>()
            })
            .map_err(|e| e.to_string());
        reset_all();
        got
    })
    .join()
    .unwrap();
    assert!(
        fired.load(Ordering::SeqCst),
        "the renames ran mid-statement"
    );
    // u.x is now the 20 that was y: the statement reads that, or refuses
    match &got {
        Ok(ids) => assert_eq!(ids, &[1], "read with the old layout"),
        Err(e) => assert!(e.contains("schema changed"), "{e}"),
    }
}

#[test]
fn a_where_value_no_joined_row_reaches_is_never_read() {
    let _guard = serial();
    let db = Database::open("memory://join_nested_sides_unreached").unwrap();
    for sql in [
        "CREATE TABLE e (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 7), (2, 8)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "INSERT INTO r VALUES (1), (2)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO m VALUES (1, -9223372036854775808)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, rows) in [
        (
            "SELECT e.id FROM e LEFT JOIN u ON u.id = e.id WHERE u.x > (SELECT ABS(m.x) FROM m)",
            0,
        ),
        (
            "SELECT e.id FROM e LEFT JOIN u ON u.id = e.id JOIN r ON r.id = e.a \
             WHERE u.x > (SELECT ABS(m.x) FROM m)",
            0,
        ),
        (
            "SELECT u.id FROM u JOIN r ON r.id = u.id \
             WHERE u.id > 5 AND u.x + r.id IN (SELECT ABS(m.x) FROM m)",
            0,
        ),
        (
            "SELECT u.id FROM u JOIN r ON r.id = u.id JOIN r AS q ON q.id = r.id \
             WHERE u.id > 5 AND u.x + q.id IN (SELECT ABS(m.x) FROM m)",
            0,
        ),
    ] {
        let got = answer(&db, sql).map(|found| found.len());
        if got != Ok(rows) {
            wrong.push(format!("{sql}: {got:?}, want {rows} rows"));
        }
        if prepared_line(&db, sql) != "Prepared: yes" {
            wrong.push(format!("{sql}: not prepared"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
