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

//! A grouped join whose clauses hold no subquery feeds its calls through
//! programs its plan holds: it answers as the general path does, and run
//! again compiles nothing through the evaluator bridge

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{bridge_compiles, PREPARED_FAMILIES_OFF};
use stoolap::{Database, Value};

// The families switch and the bridge counter are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The column names and each row, or the error
type Answer = Result<(Vec<String>, Vec<String>), String>;

fn answer(db: &Database, sql: &str, params: &[Value]) -> Answer {
    let result = db.query(sql, params.to_vec()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
    let mut rows = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        rows.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok((columns, rows))
}

/// `sql` answered with the prepared families refused: the general path
fn general(db: &Database, sql: &str, params: &[Value]) -> Answer {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            PREPARED_FAMILIES_OFF.store(false, SeqCst);
        }
    }
    let _restore = Restore;
    PREPARED_FAMILIES_OFF.store(true, SeqCst);
    answer(db, sql, params)
}

fn open(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for i in 1..=40 {
        db.execute("INSERT INTO l VALUES ($1, $2, $3)", (i, i % 7, i % 4))
            .unwrap();
    }
    for i in 1..=30 {
        db.execute("INSERT INTO r VALUES ($1, $2, $3)", (i, i % 5, i * 3))
            .unwrap();
    }
    db
}

/// Each statement with the parameter values of two runs
const SHAPES: &[(&str, &[i64], &[i64])] = &[
    (
        "SELECT l.id, SUM(r.w + 1) FROM l LEFT JOIN r ON l.k = r.k GROUP BY l.id ORDER BY l.id",
        &[],
        &[],
    ),
    (
        "SELECT 10 - l.id AS id, COUNT(*) FROM l JOIN r ON r.k = l.k GROUP BY 10 - l.id \
         ORDER BY 1",
        &[],
        &[],
    ),
    (
        "SELECT COUNT(*), SUM(CASE WHEN r.id IS NULL THEN 1 ELSE 0 END) FROM l LEFT JOIN r \
         ON l.k = r.k",
        &[],
        &[],
    ),
    (
        "SELECT l.g, SUM(r.w * l.k) FROM l JOIN r ON r.k = l.k GROUP BY l.g \
         HAVING SUM(r.w) > 10 ORDER BY l.g",
        &[],
        &[],
    ),
    (
        "SELECT l.g, SUM(r.w + $1) FROM l JOIN r ON r.k = l.k GROUP BY l.g ORDER BY l.g",
        &[1],
        &[100],
    ),
    (
        "SELECT l.g, MAX(r.w - l.id) FROM l JOIN r ON r.k = l.k WHERE l.id > $1 GROUP BY l.g \
         ORDER BY l.g",
        &[5],
        &[30],
    ),
    (
        "SELECT l.g, SUM(r.w + 1) + ROW_NUMBER() OVER (ORDER BY l.g) FROM l JOIN r \
         ON r.k = l.k GROUP BY l.g ORDER BY l.g",
        &[],
        &[],
    ),
    (
        "SELECT d.g, SUM(d.w + 1) FROM (SELECT l.g, r.w FROM l JOIN r ON r.k = l.k) AS d \
         GROUP BY d.g ORDER BY d.g",
        &[],
        &[],
    ),
];

fn values(params: &[i64]) -> Vec<Value> {
    params.iter().map(|&p| Value::Integer(p)).collect()
}

#[test]
fn a_grouped_join_answers_as_the_general_path_run_after_run() {
    let _guard = serial();
    let db = open("memory://join_grouped_feed_answers");
    let mut wrong = Vec::new();
    for (sql, first, second) in SHAPES {
        for params in [first, second, first] {
            let params = values(params);
            let (got, want) = (answer(&db, sql, &params), general(&db, sql, &params));
            if got != want || got.is_err() {
                wrong.push(format!("{sql} {params:?}\n  got  {got:?}\n  want {want:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_grouped_join_run_again_compiles_nothing_through_the_bridge() {
    let _guard = serial();
    let db = open("memory://join_grouped_feed_bridge");
    let mut wrong = Vec::new();
    for (sql, first, second) in SHAPES {
        answer(&db, sql, &values(first)).unwrap();
        let before = bridge_compiles();
        answer(&db, sql, &values(second)).unwrap();
        let compiled = bridge_compiles() - before;
        if compiled != 0 {
            wrong.push(format!("{sql}: run again, {compiled} bridge compiles"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
