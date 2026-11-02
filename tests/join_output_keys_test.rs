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

//! A join holding a subquery anywhere projects its output through the plan:
//! its computed items and keys run on no general path, it answers as the
//! join without the subquery does, and a bare name a merged pair and another
//! table both carry is ambiguous in every clause

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
    "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
    "INSERT INTO u VALUES (1, 1, 5), (2, 2, 7), (3, 1, NULL), (4, 4, 1), (5, 2, 3)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO w VALUES (1, 1), (2, 2), (3, 9)",
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

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn check(db: &Database, mode: &str) -> Vec<String> {
    let mut wrong = Vec::new();
    // `{w}` is a WHERE condition holding a subquery, or the same as a value
    for shape in [
        "SELECT t.id, u.v FROM t JOIN u ON u.k = t.k WHERE {w} ORDER BY u.v * 2, t.id, u.id",
        "SELECT t.id, u.v FROM t JOIN u ON u.k = t.k WHERE {w} ORDER BY u.v * 2, t.id, u.id LIMIT 3",
        "SELECT t.k % 2 AS m, u.v FROM t JOIN u ON u.k = t.k WHERE {w} ORDER BY m, u.v, t.id",
        "SELECT t.a AS z, u.v FROM t JOIN u ON u.k = t.k WHERE {w} ORDER BY z + 0, t.id, u.id",
        "SELECT DISTINCT ON (t.k % 2) t.k % 2, u.v FROM t JOIN u ON u.k = t.k WHERE {w} \
         ORDER BY t.k % 2, u.v DESC",
        "SELECT DISTINCT ON (1) t.k % 2, u.v FROM t JOIN u ON u.k = t.k WHERE {w} \
         ORDER BY 1, u.v",
        "SELECT t.id + u.id, COALESCE(u.v, 0) FROM t LEFT JOIN u ON u.id = t.a / 10 WHERE {w} \
         ORDER BY 2 DESC, 1",
        "SELECT k, t.id, u.v FROM t JOIN u USING (k) WHERE {w} ORDER BY k * 10 + u.v, t.id",
        "SELECT t.k % 2 FROM t JOIN u ON u.k = t.k WHERE {w} UNION ALL SELECT 5 ORDER BY 1",
        "SELECT DISTINCT ON (1) t.k % 2 FROM t JOIN u ON u.k = t.k WHERE {w} UNION ALL SELECT 5",
    ] {
        let sql = shape.replace("{w}", "t.id > (SELECT MIN(id) FROM w) - 1");
        let twin = shape.replace("{w}", "t.id > 0");
        let line = prepared_line(db, &sql);
        if line != "Prepared: yes" {
            wrong.push(format!("{mode}: {sql}: EXPLAIN {line}"));
        }
        let before = general_path_total();
        let got = answer(db, &sql);
        if general_path_total() != before {
            wrong.push(format!("{mode}: {sql}: a general path ran"));
        }
        let want = answer(db, &twin);
        if got.is_err() || got != want {
            wrong.push(format!("{mode}: {sql}: {got:?}, want {want:?}"));
        }
    }
    // A bare name a merged pair and another table both carry is ambiguous
    let merged = "t JOIN u USING (id) JOIN w AS r ON r.x = t.k";
    for clause in [
        "SELECT id FROM {m} WHERE t.a > (SELECT 0)",
        "SELECT t.a FROM {m} WHERE id > (SELECT 0)",
        "SELECT t.a FROM {m} WHERE t.a > (SELECT 0) ORDER BY id",
        "SELECT id, COUNT(*) FROM {m} WHERE t.a > (SELECT 0) GROUP BY id",
        "SELECT t.a, (SELECT id) FROM {m}",
        "SELECT id + (SELECT 0) FROM {m}",
        "SELECT id, COUNT(*) FROM {m} GROUP BY id HAVING COUNT(*) > (SELECT 0)",
        "SELECT id, ROW_NUMBER() OVER w FROM {m} WINDOW w AS (ORDER BY t.a + (SELECT 0))",
    ] {
        let sql = clause.replace("{m}", merged);
        match answer(db, &sql) {
            Err(e) if e == "invalid argument: column reference 'id' is ambiguous" => {}
            got => wrong.push(format!("{mode}: {sql}: {got:?}, want an ambiguous column")),
        }
    }
    // A SELECT alias of that name is read as the alias; a qualified name or
    // a merge of all three reads its one column
    for (sql, want) in [
        (
            format!("SELECT t.id AS id FROM {merged} WHERE t.a > (SELECT 0) ORDER BY id"),
            vec!["1", "2", "5"],
        ),
        (
            format!("SELECT t.id, r.id FROM {merged} WHERE t.a > (SELECT 0) ORDER BY t.id"),
            vec!["1|1", "2|2", "5|2"],
        ),
        (
            "SELECT id FROM t JOIN u USING (id) JOIN w USING (id) WHERE t.a > (SELECT 0) \
             ORDER BY id"
                .to_string(),
            vec!["1", "2"],
        ),
    ] {
        let want: Vec<String> = want.into_iter().map(String::from).collect();
        if answer(db, &sql) != Ok(want.clone()) {
            wrong.push(format!(
                "{mode}: {sql}: {:?}, want {want:?}",
                answer(db, &sql)
            ));
        }
    }
    wrong
}

#[test]
fn a_join_holding_a_subquery_projects_its_output_through_the_plan() {
    let _guard = serial();
    let mut wrong = check(&open("memory://join_output_keys"), "hot");
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
