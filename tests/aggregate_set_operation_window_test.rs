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

//! A statement that aggregates its rows and is a set operation's branch or
//! computes a window by a WINDOW definition answers on the runner as on the
//! general path, rows in order, over every source, hot and sealed

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    aggregate_paths_taken, reset_aggregate_paths, PREPARED_FAMILIES_OFF,
};
use stoolap::Database;

// The families switch and the path counters are process-wide
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

/// `sql` answered with the prepared families refused: the general path
fn general(db: &Database, sql: &str) -> Answer {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            PREPARED_FAMILIES_OFF.store(false, SeqCst);
        }
    }
    let _restore = Restore;
    PREPARED_FAMILIES_OFF.store(true, SeqCst);
    rows(db, sql)
}

fn open(dsn: &str, seal: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, a INTEGER)",
        "CREATE INDEX t_g ON t(g)",
        "CREATE VIEW v AS SELECT id, g, s, a FROM t",
        "INSERT INTO t VALUES (1, 1, 'x', 1), (2, 1, 'y', 2), (3, 2, 'x', 3), (4, 3, 'y', NULL), \
         (5, 3, 'x', 5), (6, 4, 'z', 1)",
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
    "(SELECT id, g, s, a FROM t) t",
    "v t",
    "(VALUES (1, 1, 'x', 1), (2, 1, 'y', 2), (3, 2, 'x', 3), (4, 3, 'y', NULL), \
     (5, 3, 'x', 5), (6, 4, 'z', 1)) t(id, g, s, a)",
];

const SHAPES: &[&str] = &[
    "SELECT g, COUNT(*) FROM {s} GROUP BY g UNION SELECT 9, 9 ORDER BY 1",
    "SELECT g, COUNT(*) FROM {s} GROUP BY g UNION ALL SELECT g, 1 FROM {s} ORDER BY 1, 2",
    "SELECT COUNT(*) FROM {s} UNION ALL SELECT COUNT(*) FROM {s} WHERE g > 1 ORDER BY 1",
    "SELECT g, SUM(a) FROM {s} GROUP BY g HAVING COUNT(*) > 1 \
     INTERSECT SELECT g, SUM(a) FROM {s} GROUP BY g ORDER BY g",
    "SELECT g FROM {s} GROUP BY g EXCEPT SELECT 2 ORDER BY g DESC LIMIT 2",
    "SELECT g, COUNT(*) AS c FROM {s} GROUP BY g UNION SELECT 1, 99 ORDER BY c DESC LIMIT 1 OFFSET 1",
    "SELECT DISTINCT ON (s) s, COUNT(*) FROM {s} GROUP BY s UNION ALL SELECT 'q', 0 ORDER BY 1",
    "SELECT g, COUNT(*), RANK() OVER (ORDER BY COUNT(*)) FROM {s} GROUP BY g \
     UNION ALL SELECT 0, 0, 0 ORDER BY 1",
    "SELECT 1 UNION ALL SELECT g FROM {s} GROUP BY g ORDER BY 1",
    "SELECT g, SUM(a), RANK() OVER w FROM {s} GROUP BY g WINDOW w AS (ORDER BY SUM(a) DESC) \
     ORDER BY g",
    "SELECT g, COUNT(*), SUM(COUNT(*)) OVER w, ROW_NUMBER() OVER w FROM {s} GROUP BY g \
     WINDOW w AS (ORDER BY g) ORDER BY g",
    "SELECT s, g, MAX(a), LAG(MAX(a)) OVER w FROM {s} GROUP BY s, g \
     WINDOW w AS (PARTITION BY s ORDER BY g) ORDER BY s, g",
    "SELECT g, COUNT(*), RANK() OVER (w ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM {s} \
     GROUP BY g WINDOW w AS (ORDER BY g) ORDER BY g",
    "SELECT g, COUNT(*) FROM {s} GROUP BY g WINDOW w AS (ORDER BY g) ORDER BY g",
    "SELECT DISTINCT ON (g % 2) g, SUM(a), RANK() OVER w FROM {s} GROUP BY g \
     WINDOW w AS (ORDER BY g) ORDER BY g % 2, g",
    "SELECT COUNT(*), RANK() OVER w FROM {s} WINDOW w AS (ORDER BY COUNT(*))",
    "SELECT g, COUNT(*), RANK() OVER w FROM {s} GROUP BY g WINDOW w AS (ORDER BY g) \
     UNION ALL SELECT 0, 0, 0 ORDER BY 1",
    "SELECT g, COUNT(*), RANK() OVER w FROM {s} GROUP BY g WINDOW w AS (ORDER BY MAX(s), g)",
    "SELECT g, COUNT(*), ROW_NUMBER() OVER w FROM {s} GROUP BY g \
     WINDOW w AS (PARTITION BY g % 2 ORDER BY MIN(a))",
    "SELECT g, COUNT(*), RANK() OVER nosuch FROM {s} GROUP BY g",
    "SELECT g, COUNT(*), RANK() OVER w FROM {s} GROUP BY g WINDOW w AS (ORDER BY a)",
];

#[test]
fn an_aggregate_in_a_set_operation_or_with_a_named_window_answers_as_the_general_path() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let sealed = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let mut wrong = Vec::new();
    for (name, db) in [
        (
            "hot",
            open("memory://aggregate_set_operation_window_test", false),
        ),
        ("sealed", open(&sealed, true)),
    ] {
        for source in SOURCES {
            for shape in SHAPES {
                let sql = shape.replace("{s}", source);
                reset_aggregate_paths();
                let got = rows(&db, &sql);
                let taken = aggregate_paths_taken();
                let want = general(&db, &sql);
                if got != want || taken.contains(&"general") {
                    wrong.push(format!(
                        "{name}: {sql}\n  took {taken:?}\n  got {got:?}\n  want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
