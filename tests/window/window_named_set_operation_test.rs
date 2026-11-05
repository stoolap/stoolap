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

//! A statement that does not aggregate and computes windows by a WINDOW
//! definition, or is a set operation's branch, runs on the runner over a
//! table or another source, and answers as recorded

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::prepared_selects_built;
use stoolap::Database;

use crate::oracle;

/// Each statement's column names and rows, recorded when the general path
/// gave the same
const RECORDED: oracle::Answers = answers!("window_named_set_operation");

// The build counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The column names and each row in order, or the error
type Answer = Result<(Vec<String>, Vec<String>), String>;

fn rows(db: &Database, sql: &str) -> Answer {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok((columns, out))
}

/// What is wrong with `answer` as `key`'s recorded answer, its column
/// names on the first line
fn check(key: &str, answer: Answer) -> Option<String> {
    let lines = answer.map(|(columns, rows)| {
        let names = format!("columns: {}", columns.join("|"));
        std::iter::once(names).chain(rows).collect()
    });
    oracle::check_answer(&RECORDED, key, &lines)
}

fn open(dsn: &str, seal: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, a INTEGER)",
        "CREATE INDEX t_g ON t(g)",
        "CREATE VIEW v AS SELECT id, g, a FROM t",
        "INSERT INTO t VALUES (1, 1, 1), (2, 1, 2), (3, 2, 3), (4, 3, NULL), (5, 3, 5), (6, 4, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each source of t's rows under the name t
const SOURCES: &[&str] = &[
    "t",
    "(SELECT id, g, a FROM t) t",
    "v t",
    "(VALUES (1, 1, 1), (2, 1, 2), (3, 2, 3), (4, 3, NULL), (5, 3, 5), (6, 4, 1)) t(id, g, a)",
];

const SHAPES: &[&str] = &[
    "SELECT id, ROW_NUMBER() OVER w FROM {s} WINDOW w AS (ORDER BY id) ORDER BY id",
    "SELECT id, SUM(a) OVER w, MAX(a) OVER w FROM {s} WINDOW w AS (PARTITION BY g ORDER BY id) ORDER BY id",
    "SELECT id, RANK() OVER (w ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM {s} WINDOW w AS (ORDER BY id) ORDER BY id",
    "SELECT id, LAG(a) OVER w FROM {s} WINDOW w AS (PARTITION BY g ORDER BY id) ORDER BY id LIMIT 3",
    "SELECT id, COUNT(*) OVER w FROM {s} WINDOW w AS (PARTITION BY g) ORDER BY id LIMIT 2 OFFSET 2",
    "SELECT id FROM {s} WINDOW w AS (ORDER BY id) ORDER BY id",
    "SELECT id, ROW_NUMBER() OVER w FROM {s} WINDOW w AS (ORDER BY nosuch)",
    "SELECT id, ROW_NUMBER() OVER nosuch FROM {s}",
    "SELECT DISTINCT ON (g) g, ROW_NUMBER() OVER w FROM {s} WINDOW w AS (ORDER BY id) ORDER BY g, id",
    "SELECT id, ROW_NUMBER() OVER w FROM {s} WHERE a > 1 WINDOW w AS (ORDER BY id DESC) ORDER BY id",
    "SELECT id, ROW_NUMBER() OVER w AS r FROM {s} WINDOW w AS (ORDER BY id) ORDER BY r DESC",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM {s} UNION ALL SELECT 99, 0 ORDER BY 1",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM {s} UNION ALL SELECT 99, 0 LIMIT 100",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS r FROM {s} UNION SELECT 99, 0 ORDER BY 2, 1",
    "SELECT g, MIN(id) OVER (PARTITION BY g) FROM {s} INTERSECT SELECT g, 1 FROM {s} ORDER BY 1",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM {s} EXCEPT SELECT 1, 1 ORDER BY 1 LIMIT 3",
    "SELECT id, SUM(a) OVER w FROM {s} WINDOW w AS (ORDER BY id) UNION ALL SELECT 0, 0 ORDER BY 1 LIMIT 3 OFFSET 1",
    "SELECT 0, 0 UNION ALL SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM {s} ORDER BY 1",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) FROM {s} UNION ALL SELECT 99, 0 ORDER BY 1, 2 LIMIT 100 OFFSET 2",
    "SELECT DISTINCT ON (g) g, ROW_NUMBER() OVER (ORDER BY id) FROM {s} UNION ALL SELECT 99, 0 ORDER BY 1",
    "SELECT DISTINCT ON (g % 2) t.*, g % 2 AS k, t.*, ROW_NUMBER() OVER (ORDER BY id) FROM {s} \
     UNION ALL SELECT t.*, g % 2, t.*, 0 FROM {s}",
    "SELECT DISTINCT ON (g % 2) g % 2 AS k, ROW_NUMBER() OVER (ORDER BY id) AS r FROM {s} \
     UNION ALL SELECT 0, 0",
    "SELECT DISTINCT ON (g) id, ROW_NUMBER() OVER w FROM {s} WINDOW w AS (ORDER BY id) \
     UNION ALL SELECT 0, 0",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id) + 1 FROM {s} UNION ALL SELECT 99, 0 ORDER BY 1",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM {s} EXCEPT SELECT 1, 1 EXCEPT SELECT 2, 2 \
     EXCEPT SELECT 3, 3 EXCEPT SELECT 4, 4 EXCEPT SELECT 5, 5 LIMIT 1",
    "SELECT id, ROW_NUMBER() OVER w FROM {s} WINDOW w AS (ORDER BY id) INTERSECT SELECT 6, 6 LIMIT 1",
];

#[test]
fn a_windowed_statement_answers_as_recorded() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let sealed = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let mut wrong = Vec::new();
    for (name, db) in [
        (
            "hot",
            open("memory://window_named_set_operation_test", false),
        ),
        ("sealed", open(&sealed, true)),
    ] {
        for source in SOURCES {
            for shape in SHAPES {
                let sql = shape.replace("{s}", source);
                wrong.extend(check(&format!("{name}: {sql}"), rows(&db, &sql)));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The statements `sql` prepares for the runner in one run
fn built(db: &Database, sql: &str) -> usize {
    let before = prepared_selects_built();
    db.query(sql, ()).unwrap().count();
    prepared_selects_built() - before
}

#[test]
fn a_named_window_or_a_set_operation_takes_the_runner() {
    let _guard = serial();
    let db = open("memory://window_named_set_operation_routes", false);
    let table = "EXPLAIN SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t UNION ALL SELECT 99, 0 ORDER BY 1";
    let plan: Vec<String> = db
        .query(table, ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert!(
        plan.iter()
            .any(|line| line.trim_start().starts_with("Window")),
        "{table}: {plan:?}"
    );
    for (sql, statements) in [
        (
            "SELECT id, ROW_NUMBER() OVER w FROM (SELECT id, g, a FROM t) t \
             WINDOW w AS (ORDER BY id) ORDER BY id",
            1,
        ),
        (
            "SELECT id, ROW_NUMBER() OVER w FROM v t WINDOW w AS (ORDER BY id) ORDER BY id",
            1,
        ),
        (
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM (SELECT id, g, a FROM t) t \
             UNION ALL SELECT 99, 0 ORDER BY 1",
            2,
        ),
    ] {
        assert_eq!(built(&db, sql), statements, "{sql}");
    }
}
