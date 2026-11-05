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

//! A statement that aggregates its rows and keeps the first group of each
//! DISTINCT ON key answers on the runner as recorded, rows in order, over
//! every source, hot and sealed

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{aggregate_paths_taken, reset_aggregate_paths};
use stoolap::Database;

use crate::oracle;

/// Each statement's column names and rows, recorded when the general path
/// gave the same
const RECORDED: oracle::Answers = answers!("aggregate_distinct_on");

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
    "SELECT DISTINCT ON (g) g, COUNT(*) FROM {s} GROUP BY g ORDER BY g",
    "SELECT DISTINCT ON (s) g, s, COUNT(*) FROM {s} GROUP BY g, s ORDER BY s, COUNT(*) DESC, g",
    "SELECT DISTINCT ON (s) g, COUNT(*) FROM {s} GROUP BY g, s ORDER BY s, g",
    "SELECT DISTINCT ON (COUNT(*)) g, COUNT(*) FROM {s} GROUP BY g ORDER BY COUNT(*), g",
    "SELECT DISTINCT ON (COUNT(*)) g FROM {s} GROUP BY g ORDER BY COUNT(*) DESC, g",
    "SELECT DISTINCT ON (SUM(a) > 2) g FROM {s} GROUP BY g ORDER BY SUM(a) > 2, g",
    "SELECT DISTINCT ON (SUM(a) > 2) g FROM {s} GROUP BY g",
    "SELECT DISTINCT ON (s, COUNT(*)) s, g FROM {s} GROUP BY s, g ORDER BY s, COUNT(*), g",
    "SELECT DISTINCT ON (c) g, COUNT(*) AS c FROM {s} GROUP BY g ORDER BY c, g",
    "SELECT DISTINCT ON (2) g, COUNT(*) FROM {s} GROUP BY g ORDER BY 2, g",
    "SELECT DISTINCT ON (g % 2) g FROM {s} GROUP BY g ORDER BY g",
    "SELECT DISTINCT ON (UPPER(s)) UPPER(s), COUNT(*) FROM {s} GROUP BY UPPER(s) ORDER BY UPPER(s)",
    "SELECT DISTINCT ON (COUNT(*)) COUNT(*) FROM {s}",
    "SELECT DISTINCT ON (SUM(a)) SUM(a) AS total, COUNT(*) FROM {s}",
    "SELECT DISTINCT ON (s) s, g, SUM(a) FROM {s} GROUP BY s, g ORDER BY s, g DESC LIMIT 1",
    "SELECT DISTINCT ON (s) s, g, SUM(a) FROM {s} GROUP BY s, g ORDER BY s, g DESC LIMIT 2 OFFSET 1",
    "SELECT DISTINCT ON (g % 2) g, COUNT(*), RANK() OVER (ORDER BY g) FROM {s} GROUP BY g \
     ORDER BY g % 2, g",
    "SELECT DISTINCT ON (g % 2) g, COUNT(*) FROM {s} GROUP BY g HAVING COUNT(*) > 0 \
     ORDER BY g % 2, g DESC",
    "SELECT DISTINCT ON (s) s, COUNT(*) FROM {s} WHERE a < 9 GROUP BY s ORDER BY s",
    "SELECT DISTINCT ON (MAX(a)) MAX(a), g FROM {s} GROUP BY g ORDER BY MAX(a) DESC, g",
    "SELECT DISTINCT ON (g, s) g, s, COUNT(*) FROM {s} GROUP BY ROLLUP (g, s) ORDER BY g, s",
    "SELECT DISTINCT ON (s) s, STRING_AGG(CAST(a AS TEXT), ',' ORDER BY a) FROM {s} GROUP BY s \
     ORDER BY s",
    "SELECT DISTINCT ON (nosuch) g, COUNT(*) FROM {s} GROUP BY g",
    "SELECT DISTINCT ON (a) g, COUNT(*) FROM {s} GROUP BY g",
    "SELECT DISTINCT ON (5) g, COUNT(*) FROM {s} GROUP BY g",
];

#[test]
fn an_aggregate_with_distinct_on_answers_as_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let sealed = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let mut wrong = Vec::new();
    for (name, db) in [
        ("hot", open("memory://aggregate_distinct_on_test", false)),
        ("sealed", open(&sealed, true)),
    ] {
        for source in SOURCES {
            for shape in SHAPES {
                let sql = shape.replace("{s}", source);
                reset_aggregate_paths();
                let got = rows(&db, &sql);
                let taken = aggregate_paths_taken();
                if taken.contains(&"general") {
                    wrong.push(format!("{name}: {sql}: took {taken:?}"));
                }
                wrong.extend(check(&format!("{name}: {sql}"), got));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_prepared_aggregate_with_distinct_on_reads_each_run_s_parameter() {
    let db = open("memory://aggregate_distinct_on_prepared", false);
    let sql = "SELECT DISTINCT ON (g > $1) g, COUNT(*) FROM t GROUP BY g ORDER BY g > $1, g DESC";
    let statement = db.prepare(sql).unwrap();
    for (bound, want) in [(0i64, vec![4]), (2, vec![2, 4])] {
        let got: Vec<i64> = statement
            .query((bound,))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(got, want, "{sql} with {bound}");
    }
}
