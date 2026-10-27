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

//! A prepared join projects its SELECT list, ORDER BY and DISTINCT ON keys
//! as a prepared statement over one table does, and answers, with the same
//! columns, as recorded where the general path gave the same answer

#![cfg(feature = "test-failpoints")]

mod oracle;

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{general_path_hits, GeneralPath};
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const MECHANISMS: [GeneralPath; 6] = [
    GeneralPath::ProjectionEvaluator,
    GeneralPath::OrderByEvaluator,
    GeneralPath::ScalarSubquery,
    GeneralPath::SelectFold,
    GeneralPath::CorrelatedExpression,
    GeneralPath::OuterBinding,
];

fn hits() -> Vec<usize> {
    MECHANISMS
        .iter()
        .map(|path| general_path_hits(*path))
        .collect()
}

/// NULLs, repeated keys and unmatched rows on both sides
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 4, 20), (4, NULL, 40), (5, 3, 50), (6, 1, 60)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9), (5, NULL)",
    "CREATE TABLE r (id INTEGER PRIMARY KEY, y INTEGER)",
    "INSERT INTO r VALUES (1, 3), (2, 1), (3, 2)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO w VALUES (1, 1), (2, 2), (3, 3), (4, 1)",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://join_output_projection_{name}")).unwrap();
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

const FROMS: &[&str] = &[
    "t JOIN u ON u.id = t.a",
    "t LEFT JOIN u ON u.id = t.a",
    "t RIGHT JOIN u ON u.id = t.a",
    "t FULL JOIN u ON u.id = t.a",
    "t JOIN u ON u.id = t.a JOIN r ON r.id = u.id",
];

/// Each SELECT list with the keys it is ordered by, every order total
const SHAPES: &[(&str, &[&str])] = &[
    (
        "t.id, (SELECT MAX(x) FROM w)",
        &["", " ORDER BY t.b DESC, t.id", " ORDER BY 2, 1"],
    ),
    (
        "*, (SELECT COUNT(*) FROM w WHERE w.x = t.a)",
        &["", " ORDER BY t.id, u.id"],
    ),
    (
        "t.*, (SELECT MAX(x) FROM w) AS m",
        &["", " ORDER BY t.id DESC, m"],
    ),
    (
        "u.*, t.b + (SELECT MIN(x) FROM w) AS s",
        &["", " ORDER BY s, u.id, t.id"],
    ),
    (
        "t.id AS k, (SELECT COUNT(*) FROM w WHERE w.x = u.id) AS c",
        &[
            "",
            " ORDER BY c DESC, k",
            " ORDER BY k + 1 DESC",
            " ORDER BY u.x, k",
        ],
    ),
    (
        "t.id, t.b",
        &[
            " ORDER BY (SELECT MAX(x) FROM w) - t.id, t.id",
            " ORDER BY t.b + (SELECT 0), t.id",
        ],
    ),
    (
        "t.id, t.b * $1 + (SELECT MAX(x) FROM w) AS p",
        &["", " ORDER BY p, t.id"],
    ),
    (
        "DISTINCT ON (t.b) t.b, t.id, (SELECT MAX(x) FROM w WHERE w.x <= t.id) AS m",
        &[" ORDER BY t.b, t.id DESC"],
    ),
];

const LIMITS: &[&str] = &["", " LIMIT 3", " LIMIT 2 OFFSET 1"];

/// The column names and the rows of `sql` as text, sorted unless the
/// statement orders them, or the error
fn answer(db: &Database, sql: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let rows = db.query(sql, (2,)).map_err(|e| e.to_string())?;
    let columns = rows.columns().to_vec();
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
    Ok((columns, out))
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), (2,))
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

/// Each statement's columns and rows, recorded where the general path gave
/// the same ones
const RECORDED: oracle::Answers = answers!("join_output_projection");

/// Each statement answers as recorded; prepared, it runs no general path
/// mechanism
fn check(db: &Database, context: &str, wrong: &mut Vec<String>) {
    for from in FROMS {
        for (select, orders) in SHAPES {
            for order in *orders {
                let limits = match order.is_empty() {
                    true => &LIMITS[..1],
                    false => LIMITS,
                };
                for limit in limits {
                    let sql = format!("SELECT {select} FROM {from}{order}{limit}");
                    let before = hits();
                    let got = answer(db, &sql);
                    let ran: Vec<_> = MECHANISMS
                        .iter()
                        .zip(before.iter().zip(hits()))
                        .filter(|(_, (before, now))| now > before)
                        .map(|(path, _)| *path)
                        .collect();
                    let got = got.map(|(columns, rows)| {
                        std::iter::once(columns.join("|")).chain(rows).collect()
                    });
                    wrong.extend(
                        oracle::check_answer(&RECORDED, &sql, &got)
                            .map(|why| format!("{context}: {why}")),
                    );
                    let line = prepared_line(db, &sql);
                    if line != "Prepared: yes" || !ran.is_empty() {
                        wrong.push(format!("{context}: {sql}\n  {line}, ran {ran:?}"));
                    }
                }
            }
        }
    }
}

#[test]
fn a_prepared_join_projects_as_a_prepared_statement_and_answers_as_the_general_path() {
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
