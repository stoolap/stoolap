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

//! A grouped statement's ORDER BY key computed over its groups is projected
//! with the SELECT items, so the sort reads it as a column

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{general_path_hits, GeneralPath};
use stoolap::Database;

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => "memory://grouped_computed_order_keys".to_string(),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER, s TEXT)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO t VALUES (1, 30, 1, 'x'), (2, 10, 2, 'y'), (3, NULL, 1, 'x'), (4, 20, 3, NULL), (5, 10, 2, 'z')",
        "INSERT INTO u VALUES (1, 1, 5), (2, 2, 7), (3, 1, NULL), (4, 4, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each row's values but the last `drop`
fn rows(db: &Database, sql: &str, drop: usize) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for row in db.query(sql, ()).map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len() - drop)
                .map(|i| format!("{:?}", row.get::<stoolap::Value>(i).unwrap()))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    Ok(out)
}

#[test]
fn a_computed_grouped_key_sorts_as_the_same_key_written_as_an_item() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        let mode = if sealed { "sealed" } else { "hot" };
        // Each statement, and its twin with the key as a last item it sorts by
        for (sql, twin) in [
            (
                "SELECT k FROM u GROUP BY k ORDER BY ABS(MIN(v))",
                "SELECT k, ABS(MIN(v)) AS o FROM u GROUP BY k ORDER BY o",
            ),
            (
                "SELECT k FROM u GROUP BY k ORDER BY MAX(v) + 1 DESC",
                "SELECT k, MAX(v) + 1 AS o FROM u GROUP BY k ORDER BY o DESC",
            ),
            (
                "SELECT COUNT(*) FROM t GROUP BY k ORDER BY k + 0",
                "SELECT COUNT(*), k + 0 AS o FROM t GROUP BY k ORDER BY o",
            ),
            (
                "SELECT k, COUNT(*) FROM t GROUP BY k \
                 ORDER BY CASE WHEN k = 2 THEN 0 ELSE COUNT(*) END, k",
                "SELECT k, COUNT(*), CASE WHEN k = 2 THEN 0 ELSE COUNT(*) END AS o FROM t \
                 GROUP BY k ORDER BY o, k",
            ),
            (
                "SELECT k, MAX(a) FROM t GROUP BY k ORDER BY MAX(a) * -1 NULLS FIRST LIMIT 2",
                "SELECT k, MAX(a), MAX(a) * -1 AS o FROM t GROUP BY k ORDER BY o NULLS FIRST LIMIT 2",
            ),
            (
                "SELECT k, MAX(a) FROM t GROUP BY k HAVING COUNT(*) > 0 ORDER BY MAX(a) - MIN(a), k",
                "SELECT k, MAX(a), MAX(a) - MIN(a) AS o FROM t GROUP BY k HAVING COUNT(*) > 0 \
                 ORDER BY o, k",
            ),
            (
                "SELECT t.k, COUNT(*) FROM t JOIN u ON u.k = t.k GROUP BY t.k \
                 ORDER BY SUM(u.v) + 0, t.k",
                "SELECT t.k, COUNT(*), SUM(u.v) + 0 AS o FROM t JOIN u ON u.k = t.k GROUP BY t.k \
                 ORDER BY o, t.k",
            ),
            (
                "SELECT k, COUNT(*) FROM t WHERE a > (SELECT 0) GROUP BY k \
                 ORDER BY COUNT(*) * 10 + k",
                "SELECT k, COUNT(*), COUNT(*) * 10 + k AS o FROM t WHERE a > (SELECT 0) \
                 GROUP BY k ORDER BY o",
            ),
            (
                "SELECT SUM(a) FROM t ORDER BY SUM(a) + 1",
                "SELECT SUM(a), SUM(a) + 1 AS o FROM t ORDER BY o",
            ),
            (
                "SELECT k, STRING_AGG(s, ',') AS g FROM t GROUP BY k \
                 ORDER BY LENGTH(STRING_AGG(s, ',')), k",
                "SELECT k, STRING_AGG(s, ',') AS g, LENGTH(STRING_AGG(s, ',')) AS o FROM t \
                 GROUP BY k ORDER BY o, k",
            ),
        ] {
            let before = general_path_hits(GeneralPath::OrderByEvaluator);
            let got = rows(&db, sql, 0);
            if general_path_hits(GeneralPath::OrderByEvaluator) != before {
                wrong.push(format!("{mode}: {sql} sorted through the general evaluator"));
            }
            let want = rows(&db, twin, 1);
            if got != want {
                wrong.push(format!("{mode}: {sql}\n  got {got:?}\n  want {want:?}"));
            }
        }
        // A key reading an alias, or under ROLLUP where a group key reads NULL, stays with the sort
        for (sql, want) in [
            (
                "SELECT k, SUM(a) + 1 AS x FROM t GROUP BY k ORDER BY CAST(x AS INTEGER), k",
                vec![
                    "Integer(2),Integer(21)",
                    "Integer(3),Integer(21)",
                    "Integer(1),Integer(31)",
                ],
            ),
            (
                "SELECT k, MAX(a) AS m FROM t GROUP BY k ORDER BY m * 2, k LIMIT 1 OFFSET 1",
                vec!["Integer(3),Integer(20)"],
            ),
            (
                "SELECT k AS x, COUNT(*) FROM t WHERE k < 3 GROUP BY ROLLUP (k, k + 1) \
                 ORDER BY x + 1 NULLS LAST, x",
                vec![
                    "Integer(1),Integer(2)",
                    "Integer(1),Integer(2)",
                    "Integer(2),Integer(2)",
                    "Integer(2),Integer(2)",
                    "Null(Null),Integer(4)",
                ],
            ),
            (
                "SELECT k, COUNT(*) FROM t WHERE k < 3 GROUP BY ROLLUP (k, k + 1) \
                 ORDER BY k + 1 NULLS LAST, k",
                vec![
                    "Integer(1),Integer(2)",
                    "Integer(1),Integer(2)",
                    "Integer(2),Integer(2)",
                    "Integer(2),Integer(2)",
                    "Null(Null),Integer(4)",
                ],
            ),
        ] {
            let got = rows(&db, sql, 0);
            if got != Ok(want.iter().map(|r| r.to_string()).collect()) {
                wrong.push(format!("{mode}: {sql}\n  got {got:?}\n  want {want:?}"));
            }
        }
        // A key's error is the statement's, as before
        for (sql, error) in [
            (
                "SELECT k FROM t GROUP BY k ORDER BY a + 0",
                "Column 'a' not found",
            ),
            (
                "SELECT k FROM t GROUP BY k ORDER BY nofn(k)",
                "Function not found: NOFN",
            ),
            (
                "SELECT k FROM t GROUP BY k ORDER BY MAX(a) * 9223372036854775807",
                "Integer overflow",
            ),
            // `s` is a column of t, which the alias does not hide inside a key
            (
                "SELECT k, SUM(a) AS s FROM t GROUP BY k ORDER BY s + 1, k",
                "Column 's' not found",
            ),
            (
                "SELECT k, COUNT(*) AS x FROM t GROUP BY k ORDER BY t.x + 0",
                "Column 't.x' not found",
            ),
        ] {
            match rows(&db, sql, 0) {
                Err(e) if e.contains(error) => {}
                other => wrong.push(format!("{mode}: {sql}: {other:?}, want {error}")),
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}
