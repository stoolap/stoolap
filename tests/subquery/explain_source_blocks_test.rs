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

//! A statement that asks blocks over a derived table, a VALUES list, a table
//! function, no FROM, a view or a CTE prints in EXPLAIN the plan its run
//! reads, hot and sealed

#![cfg(feature = "test-failpoints")]

use stoolap::{test_failpoints, Database};

fn open(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://explain_source_blocks_{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER)",
        "CREATE INDEX t_v ON t (v)",
        "INSERT INTO t SELECT g.value, g.value, g.value % 7 FROM generate_series(1, 2000) g",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO m VALUES (1, 1990), (2, 50)",
        "CREATE VIEW tv AS SELECT id, v FROM t WHERE w = 3",
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

/// The first column of every row `sql` answers, sorted
fn firsts(db: &Database, sql: &str) -> Vec<i64> {
    let mut values: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    values.sort_unstable();
    values
}

const TOP: &[i64] = &[1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000];

/// Each statement, the first column of its rows, lines its plan shows, the
/// first the one it reads its source by, and the table read its run takes
/// when it names one
#[allow(clippy::type_complexity)]
const SHAPES: [(&str, &[i64], &[&str], Option<&str>); 12] = [
    (
        "SELECT d.id FROM (SELECT id, v FROM t WHERE v > 1900) d WHERE d.v > (SELECT MAX(x) FROM m)",
        TOP,
        &["  Derived d (d.v > (SELECT MAX(x) FROM m))"],
        None,
    ),
    (
        "SELECT column1 FROM (VALUES (1), (50), (2000)) v WHERE column1 IN (SELECT x FROM m)",
        &[50],
        &["  Values v column1 IN (SELECT x FROM m)"],
        None,
    ),
    ("SELECT (SELECT MAX(x) FROM m)", &[1990], &["  OneRow"], None),
    (
        "SELECT g.value FROM generate_series(1, 10) g WHERE g.value < (SELECT COUNT(*) FROM m) + 3",
        &[1, 2, 3, 4],
        &["  Function generate_series (g.value < ((SELECT COUNT(*) FROM m) + 3))"],
        None,
    ),
    (
        "SELECT id FROM tv WHERE v > (SELECT MAX(x) FROM m)",
        &[1991, 1998],
        &["  View tv (v > (SELECT MAX(x) FROM m))", "      SeqScan t (w = 3)"],
        None,
    ),
    (
        "WITH c AS (SELECT id, v FROM t WHERE v > 1980) SELECT id FROM c WHERE v > (SELECT MAX(x) FROM m)",
        TOP,
        &["  Derived c (v > (SELECT MAX(x) FROM m))"],
        None,
    ),
    (
        "WITH c AS (SELECT id, v FROM t WHERE v > 1980) SELECT a.id FROM c a WHERE a.v > (SELECT MIN(id) FROM c) + 15",
        &[1997, 1998, 1999, 2000],
        &["  Cte c (a.v > ((SELECT MIN(id) FROM c) + 15))"],
        None,
    ),
    (
        "WITH c AS (SELECT id, v FROM t WHERE v > 1995), e AS (SELECT x FROM m WHERE x > 10) \
         SELECT a.id FROM c a WHERE a.v > (SELECT MIN(x) FROM e) AND a.id > (SELECT MIN(id) FROM c)",
        &[1997, 1998, 1999, 2000],
        &[
            "  Cte c ((a.v > (SELECT MIN(x) FROM e)) AND (a.id > (SELECT MIN(id) FROM c)))",
            "Ctes:",
            "  e",
            "      SeqScan m (x > 10)",
        ],
        None,
    ),
    (
        "WITH c AS (SELECT x FROM m) SELECT id FROM t WHERE v > (SELECT MAX(x) FROM c)",
        TOP,
        &["  IndexScan t (v > (SELECT MAX(x) FROM c))", "Ctes:", "  c", "      SeqScan m"],
        Some("Index t_v (v range)"),
    ),
    (
        "SELECT d.id FROM (SELECT id, v FROM t WHERE v > (SELECT MAX(x) FROM m)) d",
        TOP,
        &["  Derived d", "      IndexScan t (v > (SELECT MAX(x) FROM m))"],
        Some("Index t_v (v range)"),
    ),
    (
        "SELECT COUNT(*) FROM (SELECT id, v FROM t) d WHERE d.v > (SELECT MAX(x) FROM m)",
        &[10],
        &["  Derived d (d.v > (SELECT MAX(x) FROM m))"],
        None,
    ),
    (
        "SELECT d.id FROM (SELECT id, v FROM t) d WHERE EXISTS (SELECT 1 FROM m WHERE m.x = d.v)",
        &[50, 1990],
        &["  Derived d EXISTS (SELECT 1 FROM m WHERE (m.x = d.v))"],
        None,
    ),
];

fn check(db: &Database, wrong: &mut Vec<String>, at: &str) {
    for (sql, expected, shown, read) in SHAPES {
        let plan = explain(db, sql);
        test_failpoints::take_scan_accesses();
        let answered = firsts(db, sql);
        let taken = test_failpoints::take_scan_accesses();
        if answered != expected {
            wrong.push(format!("{at} {sql}: answered {answered:?}"));
        }
        let general = ["Columns:", "Scan on", "Subquery Scan", "Values Scan"];
        let runner = plan.lines().any(|line| line.trim() == "Prepared: yes")
            && plan.lines().any(|line| line.trim() == "Subqueries:")
            && shown
                .iter()
                .all(|want| plan.lines().any(|line| line == *want));
        if !runner || general.iter().any(|text| plan.contains(text)) {
            wrong.push(format!("{at} {sql}: not the runner's plan\n{plan}"));
        }
        // The run took the plan's read of the table, and every read it took,
        // a CTE body's included, shows in the plan line by line
        if let Some(read) = read {
            let planned = |access: &String| access.split("; ").all(|line| plan.contains(line));
            let took = taken.iter().any(|access| access.contains(read));
            if !plan.contains(read) || !took || !taken.iter().all(planned) {
                wrong.push(format!("{at} {sql}: ran {taken:?}, planned\n{plan}"));
            }
        }
    }
}

#[test]
fn a_statement_asking_blocks_over_a_source_explains_the_plan_its_run_reads() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open("reads", &dir, sealed);
        check(&db, &mut wrong, if sealed { "sealed" } else { "hot" });
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n\n"));
}

#[test]
fn explain_runs_no_view_whose_names_were_not_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let db = open("unnamed_view", &dir, false);
    let mut wrong = Vec::new();
    for (at, sql) in [
        "SELECT (SELECT MAX(n) FROM rv)",
        "SELECT id FROM t WHERE v > (SELECT MAX(n) FROM rv)",
        "SELECT d.id FROM (SELECT id FROM t) d WHERE d.id IN (SELECT n FROM rv)",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM rv JOIN m ON rv.n = m.x WHERE m.x = t.v)",
        "SELECT id FROM t ORDER BY (SELECT MAX(n) FROM rv), id LIMIT 1",
    ]
    .into_iter()
    .enumerate()
    {
        // A view run once records its names, so each statement reads its own
        let view = format!("rv{at}");
        db.engine()
            .create_view(&view, "SELECT x AS n FROM m".to_string(), None, false)
            .unwrap();
        let sql = sql.replace("rv", &view);
        test_failpoints::take_scan_accesses();
        let plan = explain(&db, &sql);
        let ran = test_failpoints::take_scan_accesses();
        if !ran.is_empty() {
            wrong.push(format!("{sql}: EXPLAIN ran {ran:?}\n{plan}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n\n"));
}

#[test]
fn an_inner_with_keeps_the_outer_ctes_a_table_shares_a_name_with() {
    let dir = tempfile::tempdir().unwrap();
    let db = open("inner_with", &dir, false);
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX c_x ON c (x)", ()).unwrap();
    db.execute("INSERT INTO c VALUES (1, 1), (2, 2)", ())
        .unwrap();
    let sql = "WITH c(x) AS (SELECT v FROM t WHERE v > 1995) SELECT d.x FROM \
               (WITH e(y) AS (SELECT 1997) SELECT x FROM c WHERE x > (SELECT y FROM e)) d";
    assert_eq!(firsts(&db, sql), [1998, 1999, 2000]);
    let plan = explain(&db, sql);
    assert!(
        !plan.contains("c_x") && plan.contains("Cte c (x > (SELECT y FROM e))"),
        "the plan reads the outer CTE c, not the table c\n{plan}"
    );
}
