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

//! A statement over one table that asks blocks prints in EXPLAIN the read
//! its run takes, hot and sealed

#![cfg(feature = "test-failpoints")]

use stoolap::{test_failpoints, Database};

fn open(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://explain_blocks_read_{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER)",
        "CREATE INDEX t_v ON t (v)",
        "INSERT INTO t SELECT g.value, g.value, g.value % 7 FROM generate_series(1, 2000) g",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO m VALUES (1, 1990), (2, 50)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn explain(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The first column of every row `sql` answers, in order
fn firsts(db: &Database, sql: &str) -> Vec<i64> {
    let mut values: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    values.sort_unstable();
    values
}

/// Each statement, the first column of its rows, and the read its plan
/// names: an index read the run takes, or None for a read each run picks
/// over the sets its blocks give
#[allow(clippy::type_complexity)]
const SHAPES: [(&str, &[i64], Option<&str>); 11] = [
    (
        "SELECT id FROM t WHERE v > (SELECT MAX(x) FROM m)",
        &[1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000],
        Some("Index t_v (v range)"),
    ),
    (
        "SELECT id FROM t WHERE v > (SELECT MAX(x) FROM m) AND w = 3",
        &[1991, 1998],
        Some("Index t_v (v range)"),
    ),
    (
        "SELECT COUNT(*) FROM t WHERE v > (SELECT MAX(x) FROM m)",
        &[10],
        Some("Index t_v (v range)"),
    ),
    (
        "SELECT id FROM t WHERE id = (SELECT MIN(x) FROM m)",
        &[50],
        Some("PkKey"),
    ),
    (
        "SELECT id, (SELECT MAX(x) FROM m) FROM t WHERE v > 1995",
        &[1996, 1997, 1998, 1999, 2000],
        Some("Index t_v (v range)"),
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM m WHERE x = 50) AND v > 1995",
        &[1996, 1997, 1998, 1999, 2000],
        Some("Index t_v (v range)"),
    ),
    (
        "SELECT id, (SELECT MAX(x) FROM m) FROM t WHERE v = 2000 OR w = -1 LIMIT 1",
        &[2000],
        Some("MixedOr"),
    ),
    (
        "SELECT id FROM t WHERE v = 2000 OR w = -1 LIMIT 1",
        &[2000],
        Some("MixedOr"),
    ),
    (
        "SELECT id FROM t WHERE v > ALL (SELECT x FROM m)",
        &[1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000],
        None,
    ),
    (
        "SELECT id FROM t WHERE v IN (SELECT x FROM m)",
        &[50, 1990],
        None,
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM m WHERE m.x = t.v)",
        &[50, 1990],
        None,
    ),
];

fn check(db: &Database, wrong: &mut Vec<String>, at: &str) {
    for (sql, expected, read) in SHAPES {
        let plan = explain(db, sql);
        test_failpoints::take_scan_accesses();
        let answered = firsts(db, sql);
        let taken = test_failpoints::take_scan_accesses();
        if answered != expected {
            wrong.push(format!("{at} {sql}: answered {answered:?}"));
        }
        // A statement without a subquery prints its plan with no Prepared line
        let asks = sql.contains("(SELECT");
        if plan.contains("Seq Scan") || asks != plan.lines().any(|line| line == "Prepared: yes") {
            wrong.push(format!("{at} {sql}: not the prepared plan\n{plan}"));
        }
        match read {
            Some(read) => {
                // Every read the run took is the plan's, each of its lines shown
                let shown = |access: &String| {
                    access.contains(read) && access.split("; ").all(|line| plan.contains(line))
                };
                if !plan.contains(read) || taken.is_empty() || !taken.iter().all(shown) {
                    wrong.push(format!("{at} {sql}: ran {taken:?}, planned\n{plan}"));
                }
            }
            None if !plan.contains("read chosen per run") => {
                wrong.push(format!("{at} {sql}: no per-run read\n{plan}"));
            }
            None => {}
        }
    }
}

#[test]
fn a_statement_asking_blocks_explains_the_read_its_run_takes() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open("reads", &dir, sealed);
        check(&db, &mut wrong, if sealed { "sealed" } else { "hot" });
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n\n"));
}
