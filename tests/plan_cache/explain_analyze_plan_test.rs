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

//! EXPLAIN ANALYZE prints the plan EXPLAIN prints, with what the run measured

use stoolap::Database;

fn setup(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE INDEX t_x ON t (x)",
        "CREATE INDEX u_tid ON u (tid)",
        "INSERT INTO t SELECT g.value, g.value % 10, g.value % 7 FROM generate_series(1, 2000) g",
        "INSERT INTO u SELECT g.value, g.value % 500 FROM generate_series(1, 2000) g",
        "INSERT INTO w SELECT g.value, g.value % 5 FROM generate_series(1, 50) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn lines(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect()
}

/// EXPLAIN's lines of `sql`, then EXPLAIN ANALYZE's top line with its
/// actuals cut off, and the rest of its lines but its `Paths:` line; then
/// its rows and that line
fn both(db: &Database, sql: &str) -> (Vec<String>, Vec<String>, String, String) {
    let planned = lines(db, &format!("EXPLAIN {sql}"));
    let mut analyzed = lines(db, &format!("EXPLAIN ANALYZE {sql}"));
    let top = &analyzed[0];
    let at = top.rfind(" (actual time=").unwrap_or(top.len());
    let rows = top[at..]
        .rsplit_once(", rows=")
        .map(|(_, rows)| rows.trim_end_matches(')').to_string())
        .unwrap_or_default();
    analyzed[0].truncate(at);
    let paths = match analyzed.last() {
        Some(last) if last.starts_with("Paths: ") => analyzed.pop().unwrap_or_default(),
        _ => String::new(),
    };
    (planned, analyzed, rows, paths)
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn explain_analyze_prints_the_plan_explain_prints_with_the_rows_and_paths_of_the_run() {
    let db = setup("memory://explain_analyze_plan_lines");
    let cases: &[(&str, &str, &str)] = &[
        ("SELECT id FROM t WHERE x = 3", "200", "scan.index"),
        (
            "SELECT id FROM t WHERE x > 3 ORDER BY x LIMIT 5",
            "5",
            "order_limit.prepared_top_k",
        ),
        ("SELECT id FROM t WHERE k = 2", "286", "scan.seq"),
        ("SELECT id + 0, x FROM t WHERE id = 1.5", "0", "scan.seq"),
        (
            "SELECT x, COUNT(*) FROM t GROUP BY x",
            "10",
            "aggregate.storage_grouped",
        ),
        (
            "SELECT * FROM (SELECT id, x FROM t WHERE x = 1) s WHERE s.id > 5",
            "199",
            "scan.index",
        ),
        (
            "SELECT t.id FROM t JOIN u ON u.tid = t.id WHERE t.x = 1",
            "200",
            "scan.index, join.index_nested_loop",
        ),
        (
            "SELECT t.id FROM t JOIN u ON u.tid = t.id JOIN w ON w.id = t.k WHERE w.k = 1",
            "572",
            "scan.seq x2, join.hash, join.index_nested_loop",
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.k)",
            "1429",
            "scan.seq x2",
        ),
        ("SELECT DISTINCT k FROM t", "7", "scan.seq, distinct.rows"),
        ("UPDATE t SET k = 1 WHERE x = 3", "200", "scan.index"),
        ("UPDATE t SET k = 2 WHERE id = 7", "1", "scan.index"),
        ("DELETE FROM u WHERE tid = 4", "4", "scan.index"),
        (
            "INSERT INTO w SELECT id + 100, k FROM t WHERE x = 9",
            "200",
            "scan.index",
        ),
    ];
    let mut wrong = Vec::new();
    for (sql, want, taken) in cases {
        let (planned, analyzed, rows, paths) = both(&db, sql);
        let want_paths = match taken.is_empty() {
            true => String::new(),
            false => format!("Paths: {taken}"),
        };
        if planned != analyzed || rows != *want || paths != want_paths {
            wrong.push(format!(
                "{sql}\n  EXPLAIN {planned:?}\n  ANALYZE {analyzed:?} rows={rows}, want {want}\n  \
                 {paths:?}, want {want_paths:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn explain_analyze_lists_only_the_paths_of_its_own_run() {
    let db = setup("memory://explain_analyze_own_paths");
    db.query("SELECT x, COUNT(*) FROM t GROUP BY x", ())
        .unwrap()
        .for_each(drop);
    let (.., paths) = both(&db, "SELECT id FROM t WHERE k = 2");
    assert_eq!(
        paths, "Paths: scan.seq",
        "the grouped statement before it ran outside the trace"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn explain_analyze_lists_a_filter_judged_on_the_thread_pool() {
    let db = setup("memory://explain_analyze_parallel_filter");
    db.execute(
        "INSERT INTO t SELECT g.value, g.value % 10, g.value % 7 \
         FROM generate_series(2001, 20000) g",
        (),
    )
    .unwrap();
    let (.., paths) = both(&db, "SELECT id FROM t WHERE k + 0 = 2 ORDER BY x");
    assert_eq!(
        paths, "Paths: scan.seq, scan.parallel_filter",
        "20,000 rows read past a WHERE no scan takes are judged on the pool"
    );
}
