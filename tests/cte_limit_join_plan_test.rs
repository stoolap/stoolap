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

//! A join whose LIMIT cuts a grouped CTE side keeps its plan, runs its
//! subqueries as blocks and reads the CTE under the same cut the CTE path
//! reads it under

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{general_path_hits, random_draws, GeneralPath};
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The SELECT list's projection apart, which the ledger keeps as its own cell
const MECHANISMS: [GeneralPath; 5] = [
    GeneralPath::ScalarSubquery,
    GeneralPath::CorrelatedExpression,
    GeneralPath::OuterBinding,
    GeneralPath::SelectFold,
    GeneralPath::LiftedSubqueries,
];

fn hits() -> Vec<usize> {
    MECHANISMS
        .iter()
        .map(|path| general_path_hits(*path))
        .collect()
}

/// t holds 20 groups of two rows under a; u matches every fourth group
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t SELECT g.value, (g.value + 1) / 2, g.value * 10 FROM generate_series(1, 40) g",
    "CREATE INDEX t_a ON t (a)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u SELECT g.value * 4, g.value FROM generate_series(1, 5) g",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO w VALUES (1, 4), (2, 9), (3, 8)",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://cte_limit_join_plan_{name}")).unwrap();
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

/// The CTE draws once per row of t it reads, so the draws count its cut
const WITH: &str =
    "WITH c (a, n, s) AS (SELECT a, COUNT(*), SUM(b + 0 * RANDOM()) FROM t GROUP BY a) ";

/// Each join, the CTE the side an outer join keeps or either side of an
/// inner one; `{on}` is the rest of its ON
const FROMS: &[&str] = &[
    "c LEFT JOIN u ON u.id = c.a{on}",
    "u RIGHT JOIN c ON u.id = c.a{on}",
    "c JOIN u ON u.id = c.a{on}",
    "u JOIN c ON u.id = c.a{on}",
];

/// A subquery in the select list, in the ON, or both; `{v}` is the value
/// the reference writes as the literal it reads, 9
const PLACES: &[(&str, &str)] = &[
    ("c.a, c.n, u.x, {v}", ""),
    ("c.a, c.n, u.x", " AND u.x < {v}"),
    ("c.a, c.n, {v} - u.x", " AND u.x < {v}"),
];

const ORDERS: &[&str] = &["", " ORDER BY c.a", " ORDER BY c.n DESC, c.a DESC"];
const LIMITS: &[&str] = &[" LIMIT 2", " LIMIT 2 OFFSET 1", " LIMIT 7"];

/// The rows of `sql` as text, sorted unless the statement orders them
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

/// The answer of `sql`, the draws it took and the general path mechanisms
/// it ran
fn run(db: &Database, sql: &str) -> (Result<Vec<String>, String>, usize, Vec<GeneralPath>) {
    let (before, drawn) = (hits(), random_draws());
    let got = answer(db, sql);
    let drawn = random_draws() - drawn;
    let ran = MECHANISMS
        .iter()
        .zip(before.iter().zip(hits()))
        .filter(|(_, (before, now))| now > *before)
        .map(|(path, _)| *path)
        .collect();
    (got, drawn, ran)
}

/// Each statement answers as its reference, the same statement with the
/// literal its subqueries read, which the CTE path runs; it draws as many
/// times, so it reads the CTE under the same cut, and runs no general path
/// mechanism. Without an order, its rows are rows the uncut join returns
fn check(db: &Database, context: &str, wrong: &mut Vec<String>) {
    let mut fewest = usize::MAX;
    for from in FROMS {
        for (select, on) in PLACES {
            for order in ORDERS {
                for limit in LIMITS {
                    let spell = |v: &str| {
                        let from = from.replace("{on}", on);
                        format!("{WITH}SELECT {select} FROM {from}{order}{limit}").replace("{v}", v)
                    };
                    let sql = spell("(SELECT MAX(x) FROM w)");
                    let reference = spell("9");
                    let line = prepared_line(db, &sql);
                    let (got, drawn, ran) = run(db, &sql);
                    let (want, want_drawn, _) = run(db, &reference);
                    fewest = fewest.min(want_drawn);
                    let fits = match order.is_empty() {
                        true => {
                            let uncut =
                                answer(db, &reference.replace(limit, "")).unwrap_or_default();
                            let (got, want) = (
                                got.clone().unwrap_or_default(),
                                want.clone().unwrap_or_default(),
                            );
                            got.len() == want.len() && got.iter().all(|row| uncut.contains(row))
                        }
                        false => got == want,
                    };
                    if !fits || want.is_err() {
                        wrong.push(format!("{context}: {sql}\n  got {got:?}\n  want {want:?}"));
                    }
                    if drawn != want_drawn {
                        wrong.push(format!(
                            "{context}: {sql}\n  drew {drawn}, the CTE path {want_drawn}"
                        ));
                    }
                    if line != "Prepared: yes" || !ran.is_empty() {
                        wrong.push(format!("{context}: {sql}\n  {line}, ran {ran:?}"));
                    }
                }
            }
        }
    }
    if fewest >= 40 {
        wrong.push(format!(
            "{context}: the CTE path read all of t, {fewest} draws, in every statement"
        ));
    }
}

#[test]
fn a_join_whose_limit_cuts_a_cte_side_keeps_its_plan_and_the_cut() {
    let _guard = serial();
    let mut wrong = Vec::new();
    check(&hot("matrix"), "hot", &mut wrong);
    let dir = tempfile::tempdir().unwrap();
    check(&sealed(&dir), "sealed", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
