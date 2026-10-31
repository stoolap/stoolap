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

//! A grouped block whose SELECT list, HAVING or GROUP BY holds a subquery,
//! and an EXISTS whose ORDER BY, DISTINCT ON, LIMIT or OFFSET only a
//! statement reads, run prepared as statements; an EXISTS computes no
//! ORDER BY key.

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::general_path_total;
use stoolap::Database;

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// t: (id, k), u: (id, k, v), sealed into volumes when `sealed`
fn setup(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://statement_form_blocks_{}", dir.path().display()),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 4)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 1, NULL), (3, 2, 7), (4, 3, 1), (5, 3, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

type Rows = Result<Vec<Vec<String>>, String>;

fn rows(db: &Database, sql: &str) -> Rows {
    let mut out = Vec::new();
    for row in db.query(sql, ()).map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect(),
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

/// `sql` answers as `twin`, prepared, and reads no general path mechanism
fn holds(db: &Database, sql: &str, twin: &str, mode: &str, wrong: &mut Vec<String>) {
    let before = general_path_total();
    let got = rows(db, sql);
    let general = general_path_total() - before;
    let want = rows(db, twin);
    if got != want {
        wrong.push(format!("{mode}: {sql}\n  got {got:?}\n  want {want:?}"));
    }
    if general != 0 {
        wrong.push(format!(
            "{mode}: {sql} ran {general} general path mechanisms"
        ));
    }
    let line = prepared_line(db, sql);
    if line != "Prepared: yes" {
        wrong.push(format!("{mode}: {sql} says {line}"));
    }
}

/// Each grouped block with a subquery in a clause its aggregation reads,
/// and the same block with the subquery's value written in
const GROUPED: &[(&str, &str)] = &[
    (
        "SELECT MAX(v + (SELECT MIN(v) FROM u)) FROM u WHERE u.k = t.k",
        "SELECT MAX(v + 1) FROM u WHERE u.k = t.k",
    ),
    (
        "SELECT COUNT(*) FROM u WHERE u.k = t.k HAVING MAX(v) > (SELECT MIN(v) FROM u)",
        "SELECT COUNT(*) FROM u WHERE u.k = t.k HAVING MAX(v) > 1",
    ),
    (
        "SELECT COUNT(*) + (SELECT MIN(v) FROM u) FROM u WHERE u.k = t.k",
        "SELECT COUNT(*) + 1 FROM u WHERE u.k = t.k",
    ),
    (
        "SELECT SUM(v * (SELECT COUNT(*) FROM u u2 WHERE u2.k = t.k)) FROM u WHERE u.k = t.k",
        "SELECT SUM(v * CASE t.k WHEN 1 THEN 2 WHEN 2 THEN 1 WHEN 3 THEN 2 ELSE 0 END) \
         FROM u WHERE u.k = t.k",
    ),
    (
        "SELECT COUNT(*) FROM u WHERE u.k = t.k GROUP BY (SELECT 1)",
        "SELECT COUNT(*) FROM u WHERE u.k = t.k GROUP BY 1 = 1",
    ),
];

#[test]
fn a_grouped_block_with_a_subquery_in_its_aggregation_answers_in_every_site() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        let mode = if sealed { "sealed" } else { "hot" };
        for (block, twin) in GROUPED {
            let sites = [
                "SELECT id, ({b}) FROM t ORDER BY id",
                "SELECT id, CASE WHEN ({b}) > 2 THEN 'big' ELSE 'small' END FROM t ORDER BY id",
                "SELECT id FROM t WHERE EXISTS ({b}) ORDER BY id",
                "SELECT id FROM t WHERE NOT EXISTS ({b}) ORDER BY id",
                "SELECT id FROM t WHERE 2 IN ({b}) ORDER BY id",
                "SELECT id FROM t WHERE 2 NOT IN ({b}) ORDER BY id",
                "SELECT k, COUNT(*) FROM t GROUP BY k HAVING COUNT(*) <= ({b}) ORDER BY k",
            ];
            for site in sites {
                let sql = site.replace("{b}", block);
                let literal = site.replace("{b}", twin);
                holds(&db, &sql, &literal, mode, &mut wrong);
            }
        }
        let sql = "SELECT id, (SELECT COUNT(*) FROM u WHERE u.k = t.k \
                   HAVING MAX(v) > (SELECT $1)) FROM t ORDER BY id";
        let read = |n: i64| -> Vec<Option<i64>> {
            db.query(sql, (n,))
                .unwrap()
                .map(|row| row.unwrap().get(1).unwrap())
                .collect()
        };
        for (n, want) in [
            (1, [Some(2), Some(1), Some(2), None]),
            (5, [None, Some(1), None, None]),
        ] {
            if read(n) != want {
                wrong.push(format!(
                    "{mode}: {sql} with {n}: {:?}, want {want:?}",
                    read(n)
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn an_exists_computes_no_order_by_key() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        let mode = if sealed { "sealed" } else { "hot" };
        // A key reading two rows fails wherever it is computed
        let key = "(SELECT v FROM u)";
        for (block, twin) in [
            (
                format!("SELECT 1 FROM u WHERE u.k = t.k ORDER BY {key} LIMIT 1"),
                "SELECT 1 FROM u WHERE u.k = t.k",
            ),
            (
                format!("SELECT 1 FROM u WHERE u.k = t.k ORDER BY {key}"),
                "SELECT 1 FROM u WHERE u.k = t.k",
            ),
            (
                format!("SELECT k FROM u WHERE u.k = t.k GROUP BY k ORDER BY {key} LIMIT 1"),
                "SELECT k FROM u WHERE u.k = t.k GROUP BY k",
            ),
            (
                format!("SELECT 1 FROM u WHERE u.k = t.k ORDER BY {key} LIMIT 1 OFFSET 1"),
                "SELECT 1 FROM u WHERE u.k = t.k AND u.k <> 2",
            ),
            (
                format!(
                    "SELECT 1 FROM u JOIN t t2 ON t2.id = u.id WHERE u.k = t.k \
                     ORDER BY {key} LIMIT 1"
                ),
                "SELECT 1 FROM u JOIN t t2 ON t2.id = u.id WHERE u.k = t.k",
            ),
            (
                "SELECT k FROM u WHERE u.k = t.k GROUP BY k LIMIT 0".to_string(),
                "SELECT 1 FROM u WHERE 1 = 0",
            ),
            (
                "SELECT DISTINCT ON ((SELECT 1)) k FROM u WHERE u.k = t.k OFFSET 1".to_string(),
                "SELECT 1 FROM u WHERE 1 = 0",
            ),
        ] {
            let sql = format!("SELECT id FROM t WHERE EXISTS ({block}) ORDER BY id");
            let literal = format!("SELECT id FROM t WHERE EXISTS ({twin}) ORDER BY id");
            holds(&db, &sql, &literal, mode, &mut wrong);
        }
        let unknown =
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k ORDER BY nosuch LIMIT 1)";
        if !rows(&db, unknown).is_err_and(|e| e.contains("nosuch")) {
            wrong.push(format!(
                "{mode}: {unknown} binds its key: {:?}",
                rows(&db, unknown)
            ));
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn an_exists_ordered_by_position_over_a_merged_star_checks_the_merged_width() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup(&dir, false);
    let mut wrong = Vec::new();
    let exists = |join: &str, order: &str| {
        format!(
            "SELECT id FROM t WHERE EXISTS (SELECT * FROM u {join} WHERE u.k = t.k {order}) \
             ORDER BY id"
        )
    };
    let twin = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.k) ORDER BY id";
    for (join, order) in [
        ("JOIN u u2 USING (k)", "ORDER BY 1 LIMIT 1"),
        ("JOIN u u2 USING (k)", "ORDER BY 5 LIMIT 1"),
        ("NATURAL JOIN u u2", "ORDER BY 3"),
    ] {
        holds(&db, &exists(join, order), twin, "hot", &mut wrong);
    }
    for (join, order, position) in [
        ("JOIN u u2 USING (k)", "ORDER BY 6 LIMIT 1", 6),
        ("NATURAL JOIN u u2", "ORDER BY 4", 4),
    ] {
        let sql = exists(join, order);
        let want = format!("ORDER BY position {position} is not in the select list");
        if !rows(&db, &sql).is_err_and(|e| e.contains(&want)) {
            wrong.push(format!("{sql}: {:?}, want {want}", rows(&db, &sql)));
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}
