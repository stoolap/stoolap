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

//! A window function on the left of IN, ANY or ALL over a subquery runs as
//! the windows run anywhere else in the SELECT list: the statement is
//! prepared, runs no general path, and answers as the same test over the
//! window's value read from a derived table

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::general_path_total;
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER)",
    "INSERT INTO t VALUES (1, 30, 1), (2, 10, 2), (3, NULL, 1), (4, 20, 3), (5, 10, 2)",
    "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO m VALUES (1, 1), (2, 3)",
];

fn open(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The rows of `sql` in its order, or the error
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get::<stoolap::Value>(i) {
                    Ok(value) => format!("{value:?}"),
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

/// Each statement and its twin, which reads the window's value as `w` from
/// a derived table `d` of `{rows}`
const SHAPES: &[(&str, &str)] = &[
    (
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) IN (SELECT x FROM m) FROM t ORDER BY id",
        "SELECT id, w IN (SELECT x FROM m) FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) NOT IN (SELECT x FROM m) FROM t ORDER BY id",
        "SELECT id, w NOT IN (SELECT x FROM m) FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, CASE WHEN ROW_NUMBER() OVER (ORDER BY id) IN (SELECT x FROM m) THEN 1 ELSE 0 END \
         FROM t ORDER BY id",
        "SELECT id, CASE WHEN w IN (SELECT x FROM m) THEN 1 ELSE 0 END FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, COALESCE(ROW_NUMBER() OVER (ORDER BY id) IN (SELECT x FROM m), FALSE) \
         FROM t ORDER BY id",
        "SELECT id, COALESCE(w IN (SELECT x FROM m), FALSE) FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) > ALL (SELECT x FROM m) FROM t ORDER BY id",
        "SELECT id, w > ALL (SELECT x FROM m) FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) + 0 > ANY (SELECT x FROM m) FROM t ORDER BY id",
        "SELECT id, w + 0 > ANY (SELECT x FROM m) FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) = ANY (SELECT x FROM m WHERE m.id = t.k) \
         FROM t ORDER BY id",
        "SELECT id, w = ANY (SELECT x FROM m WHERE m.id = d.k) FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, (ROW_NUMBER() OVER (ORDER BY id), id) IN (SELECT x, x FROM m) FROM t ORDER BY id",
        "SELECT id, (w, id) IN (SELECT x, x FROM m) FROM {rn} ORDER BY id",
    ),
    (
        "SELECT id, SUM(a) OVER (ORDER BY id) >= ALL (SELECT x * 10 FROM m) FROM t ORDER BY id",
        "SELECT id, w >= ALL (SELECT x * 10 FROM m) FROM \
         (SELECT id, SUM(a) OVER (ORDER BY id) AS w FROM t) d ORDER BY id",
    ),
    (
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) > ALL (SELECT x FROM m) FROM t \
         ORDER BY id LIMIT 2",
        "SELECT id, w > ALL (SELECT x FROM m) FROM \
         (SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS w FROM t) d ORDER BY id LIMIT 2",
    ),
    (
        "SELECT k, COUNT(*), RANK() OVER (ORDER BY k) > ANY (SELECT x FROM m) FROM t \
         GROUP BY k ORDER BY k",
        "SELECT k, n, w > ANY (SELECT x FROM m) FROM \
         (SELECT k, COUNT(*) AS n, RANK() OVER (ORDER BY k) AS w FROM t GROUP BY k) d ORDER BY k",
    ),
];

fn check(db: &Database, mode: &str) -> Vec<String> {
    let rn = "(SELECT id, k, ROW_NUMBER() OVER (ORDER BY id) AS w FROM t) d";
    let mut wrong = Vec::new();
    for (sql, twin) in SHAPES {
        let twin = twin.replace("{rn}", rn);
        let line = prepared_line(db, sql);
        if line != "Prepared: yes" {
            wrong.push(format!("{mode}: {sql}: EXPLAIN {line}"));
        }
        let before = general_path_total();
        let got = answer(db, sql);
        if general_path_total() != before {
            wrong.push(format!("{mode}: {sql}: a general path ran"));
        }
        let want = answer(db, &twin);
        if got.is_err() || got != want {
            wrong.push(format!("{mode}: {sql}: {got:?}, want {want:?}"));
        }
    }
    wrong
}

#[test]
fn a_window_on_the_left_of_a_quantified_test_runs_as_the_windows_do() {
    let _guard = serial();
    let mut wrong = check(&open("memory://window_quantified_left"), "hot");
    let dir = tempfile::tempdir().unwrap();
    let db = open(&format!("file://{}/db", dir.path().display()));
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    wrong.extend(check(&db, "sealed"));
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
