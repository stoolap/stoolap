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

//! A statement with window functions over one table runs on the planner's
//! plan, its rows read through the index the window rule chooses, and
//! answers as the general path does

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Mutex, MutexGuard};

use stoolap::core::Value;
use stoolap::test_failpoints::PREPARED_FAMILIES_OFF;
use stoolap::Database;

// The families switch is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// t: a indexed, b indexed, NOT NULL and distinct, so an ORDER BY b orders
/// the rows one way, c with NULLs
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL, c INTEGER)",
    "CREATE INDEX t_a ON t(a)",
    "CREATE INDEX t_b ON t(b)",
    "INSERT INTO t SELECT g.value, g.value % 7, (g.value * 37) % 1009, \
     CASE WHEN g.value % 5 = 0 THEN NULL ELSE g.value % 13 END FROM generate_series(1, 120) g",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://planned_window_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Sealed, then some rows changed and added hot
fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for sql in [
        "UPDATE t SET c = c + 1 WHERE id % 9 = 0",
        "DELETE FROM t WHERE id % 17 = 0",
        "INSERT INTO t VALUES (500, 3, 2000, 4), (501, 6, 2001, NULL)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

const WINDOWS: &[&str] = &[
    "ROW_NUMBER() OVER (PARTITION BY a ORDER BY id)",
    "RANK() OVER (PARTITION BY a ORDER BY c)",
    "SUM(c) OVER (PARTITION BY a)",
    "ROW_NUMBER() OVER (ORDER BY b)",
    "LAG(c, 2) OVER (ORDER BY b DESC)",
    "COUNT(*) OVER ()",
    "SUM(b) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)",
    "NTILE(3) OVER (ORDER BY b)",
    "DENSE_RANK() OVER (ORDER BY c NULLS FIRST)",
];

/// Each statement over `{w}`, with the parameters it reads
const SHAPES: &[(&str, &[i64])] = &[
    ("SELECT id, {w} AS x FROM t ORDER BY id", &[]),
    ("SELECT id, {w} AS x FROM t WHERE a < 5 ORDER BY id", &[]),
    (
        "SELECT id, {w} AS x FROM t WHERE c + 0 > 6 ORDER BY id",
        &[],
    ),
    ("SELECT id, {w} AS x FROM t ORDER BY x, id LIMIT 7", &[]),
    (
        "SELECT id, {w} AS x FROM t ORDER BY id LIMIT 5 OFFSET 3",
        &[],
    ),
    ("SELECT DISTINCT a, {w} AS x FROM t ORDER BY a, x", &[]),
    (
        "SELECT DISTINCT ON (a) a, id, {w} AS x FROM t ORDER BY a, id",
        &[],
    ),
    ("SELECT id FROM t ORDER BY {w}, id LIMIT 10", &[]),
    ("SELECT id, {w} + 1 AS x FROM t ORDER BY id", &[]),
    (
        "SELECT id, {w} AS x FROM t WHERE a < $1 ORDER BY id LIMIT $2",
        &[4, 9],
    ),
    ("SELECT id, {w} AS x FROM t LIMIT 6", &[]),
    (
        "SELECT DISTINCT ON ({w}) id, COUNT(*) OVER () AS n FROM t ORDER BY id",
        &[],
    ),
];

/// The rows of `sql` as text, in the order given
fn answer(db: &Database, sql: &str, params: &[i64]) -> Result<Vec<String>, String> {
    let params: Vec<Value> = params.iter().map(|&p| Value::Integer(p)).collect();
    let rows = db.query(sql, params).map_err(|e| e.to_string())?;
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

/// `run` on the general path, the prepared families off
fn general<T>(run: impl FnOnce() -> T) -> T {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            PREPARED_FAMILIES_OFF.store(false, Relaxed);
        }
    }
    PREPARED_FAMILIES_OFF.store(true, Relaxed);
    let _restore = Restore;
    run()
}

#[test]
fn a_windowed_statement_answers_as_the_general_path() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    let mut count = 0;
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir))] {
        for window in WINDOWS {
            for &(shape, params) in SHAPES {
                let sql = shape.replace("{w}", window);
                count += 1;
                let planned = answer(&db, &sql, params);
                let reference = general(|| answer(&db, &sql, params));
                // Rows a LIMIT keeps without an ORDER BY are any of the
                // statement's, each carrying its own window value
                let unordered = !shape.contains("ORDER BY") && shape.contains("LIMIT");
                let fits = match (&planned, unordered) {
                    (Ok(rows), true) => {
                        let whole = sql.split(" LIMIT").next().unwrap_or_default();
                        let all = general(|| answer(&db, whole, params)).unwrap_or_default();
                        rows.len() == reference.as_ref().map_or(0, Vec::len)
                            && rows.iter().all(|row| all.contains(row))
                    }
                    _ => planned == reference,
                };
                if !fits {
                    wrong.push(format!(
                        "{context}: {sql}\n  planned  {planned:?}\n  general  {reference:?}"
                    ));
                }
            }
        }
    }
    assert!(count > 150, "{count} statements");
    assert!(
        wrong.is_empty(),
        "{} of {count} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn explain_shows_the_window_step_and_the_index_read_it_takes() {
    let _guard = serial();
    let db = hot("explain");
    let lines = |sql: &str| -> Vec<String> {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap().trim().to_string())
            .collect()
    };
    for (sql, window, read) in [
        (
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY a ORDER BY id) FROM t LIMIT 5",
            "Window ROW_NUMBER() OVER (PARTITION BY a ORDER BY id ASC)",
            Some("PartitionsByIndex t (a) one at a time to the LIMIT"),
        ),
        (
            "SELECT id, SUM(c) OVER (PARTITION BY a) FROM t ORDER BY id",
            "Window SUM(c) OVER (PARTITION BY a)",
            Some("PartitionsByIndex t (a)"),
        ),
        (
            "SELECT id, ROW_NUMBER() OVER (ORDER BY b) FROM t LIMIT 5",
            "Window ROW_NUMBER() OVER (ORDER BY b ASC)",
            Some("IndexOrder t (b ASC) to the LIMIT"),
        ),
        (
            "SELECT id, ROW_NUMBER() OVER (ORDER BY c) FROM t LIMIT 5",
            "Window ROW_NUMBER() OVER (ORDER BY c ASC)",
            None,
        ),
        (
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY a ORDER BY id) FROM t WHERE b > 3",
            "Window ROW_NUMBER() OVER (PARTITION BY a ORDER BY id ASC)",
            None,
        ),
    ] {
        let got = lines(sql);
        assert!(got.iter().any(|l| l == window), "{sql}: {got:#?}");
        match read {
            Some(read) => {
                assert!(got.iter().any(|l| l == read), "{sql}: {got:#?}");
                assert!(
                    got.iter().any(|l| l.starts_with("else ")),
                    "{sql}: the scan the read falls back to: {got:#?}"
                );
            }
            None => assert!(
                !got.iter()
                    .any(|l| l.contains("PartitionsByIndex") || l.contains("IndexOrder")),
                "{sql}: {got:#?}"
            ),
        }
    }
}
