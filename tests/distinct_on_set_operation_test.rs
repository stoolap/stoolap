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

//! A set operation's DISTINCT ON over a first SELECT with subqueries runs
//! prepared, its keys read over the whole set operation's rows

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::general_path_total;
use stoolap::Database;

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => "memory://distinct_on_set_operation".to_string(),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO t VALUES (1, 30, 1), (2, 10, 2), (3, NULL, 1), (4, 20, 3), (5, 10, 2)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 2, 7), (3, 1, NULL), (4, 4, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for row in db.query(sql, ()).map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get::<stoolap::Value>(i).unwrap()))
                .collect::<Vec<_>>()
                .join(","),
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

#[test]
fn a_set_operation_s_distinct_on_reads_its_rows_over_a_prepared_first_select() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        let mode = if sealed { "sealed" } else { "hot" };
        for (head, tail) in [
            (
                "SELECT DISTINCT ON (k) k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY k, id",
            ),
            (
                "SELECT DISTINCT ON (k) k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY k, id DESC",
            ),
            (
                "SELECT DISTINCT ON (1) k, id",
                "UNION SELECT k, id FROM u ORDER BY 1, 2",
            ),
            (
                "SELECT DISTINCT ON (kk) k AS kk, id",
                "UNION ALL SELECT k, id FROM u ORDER BY kk, id",
            ),
            (
                "SELECT DISTINCT ON (k + 0) k + 0, id",
                "UNION ALL SELECT k, id FROM u ORDER BY 1, 2",
            ),
            (
                "SELECT DISTINCT ON (t.k) t.k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY 1, 2",
            ),
            (
                "SELECT DISTINCT ON (id) k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY id, k",
            ),
            (
                "SELECT DISTINCT ON (k) k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY k, id LIMIT 2 OFFSET 1",
            ),
            (
                "SELECT DISTINCT ON (k) k, id",
                "EXCEPT SELECT k, id FROM u ORDER BY k, id",
            ),
            (
                "SELECT DISTINCT ON (k) k, id",
                "INTERSECT SELECT k, id FROM t ORDER BY k",
            ),
            (
                "SELECT DISTINCT ON (k) *",
                "UNION ALL SELECT id, a, k FROM t ORDER BY k, id",
            ),
            (
                "SELECT DISTINCT ON (k % 2) k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY 1",
            ),
            (
                "SELECT DISTINCT ON (3) k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY 1",
            ),
            (
                "SELECT DISTINCT ON (nocol) k, id",
                "UNION ALL SELECT k, id FROM u ORDER BY 1",
            ),
        ] {
            let sql = format!("{head} FROM t WHERE a > (SELECT 1) {tail}");
            let twin = format!("{head} FROM t WHERE a > 1 {tail}");
            let before = general_path_total();
            let got = rows(&db, &sql);
            if general_path_total() != before {
                wrong.push(format!("{mode}: {sql} ran a general path mechanism"));
            }
            let want = rows(&db, &twin);
            if got != want {
                wrong.push(format!("{mode}: {sql}\n  got {got:?}\n  want {want:?}"));
            }
            let line = prepared_line(&db, &sql);
            if line != "Prepared: yes" {
                wrong.push(format!("{mode}: {sql} says {line}"));
            }
        }
        // A correlated item of the first SELECT, read once per row
        let sql = "SELECT DISTINCT ON (k) k, (SELECT MAX(v) FROM u WHERE u.k = t.k) FROM t \
                   UNION ALL SELECT k, v FROM u ORDER BY k";
        let want = [
            "Integer(1),Integer(5)",
            "Integer(2),Integer(7)",
            "Integer(3),Null(Null)",
            "Integer(4),Integer(1)",
        ];
        if rows(&db, sql) != Ok(want.iter().map(|r| r.to_string()).collect()) {
            wrong.push(format!("{mode}: {sql}: {:?}", rows(&db, sql)));
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}
