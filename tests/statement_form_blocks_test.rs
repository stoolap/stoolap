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

/// t: (id, k), u: (id, k, v), r: (rid) naming u's first rows, and w: (id, k, v)
/// with an index on k and the smallest integer, which ABS cannot negate;
/// sealed into volumes when `sealed`
fn setup(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://statement_form_blocks_{}", dir.path().display()),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "CREATE INDEX w_k ON w (k)",
        "CREATE TABLE r (rid INTEGER PRIMARY KEY)",
        "INSERT INTO r VALUES (1), (2), (3), (4)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 4)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 1, NULL), (3, 2, 7), (4, 3, 1), (5, 3, 2)",
        "INSERT INTO w VALUES (1, 1, -9223372036854775808), (2, 1, 3), (3, 2, 5)",
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
fn a_grouped_block_over_a_join_binds_its_keys_over_its_groups() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        let mode = if sealed { "sealed" } else { "hot" };
        let from = "FROM u JOIN r ON r.rid = u.id";
        for (sql, twin) in [
            (
                format!(
                    "SELECT id, (SELECT u.k {from} WHERE u.k = t.k GROUP BY u.k \
                     ORDER BY SUM(u.v) LIMIT 1) FROM t ORDER BY id"
                ),
                format!("SELECT id, (SELECT u.k {from} WHERE u.k = t.k GROUP BY u.k) FROM t ORDER BY id"),
            ),
            (
                format!(
                    "SELECT id FROM t WHERE k IN \
                     (SELECT u.k {from} GROUP BY u.k ORDER BY SUM(u.v) DESC LIMIT 1) ORDER BY id"
                ),
                "SELECT id FROM t WHERE k IN (2) ORDER BY id".to_string(),
            ),
            (
                format!("SELECT id, (SELECT COUNT(*) + (SELECT 1) {from} WHERE u.k = t.k) FROM t ORDER BY id"),
                format!("SELECT id, (SELECT COUNT(*) + 1 {from} WHERE u.k = t.k) FROM t ORDER BY id"),
            ),
            (
                format!(
                    "SELECT id, (SELECT COUNT(*) {from} WHERE u.k = t.k GROUP BY u.k \
                     HAVING COUNT(*) > (SELECT 0)) FROM t ORDER BY id"
                ),
                format!(
                    "SELECT id, (SELECT COUNT(*) {from} WHERE u.k = t.k GROUP BY u.k \
                     HAVING COUNT(*) > 0) FROM t ORDER BY id"
                ),
            ),
        ] {
            holds(&db, &sql, &twin, mode, &mut wrong);
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
        // An aggregate key reads the groups; ABS of the smallest integer fails if computed
        let groups = "SELECT k FROM w WHERE w.k = t.k GROUP BY k";
        for block in [
            format!("{groups} ORDER BY SUM(v)"),
            format!("{groups} ORDER BY SUM(v) LIMIT 1"),
            "SELECT k, SUM(v) AS total FROM w WHERE w.k = t.k GROUP BY k ORDER BY total"
                .to_string(),
            format!("{groups} ORDER BY ABS(MIN(v)) LIMIT 1"),
            "SELECT k, (SELECT 1) FROM w WHERE w.k = t.k GROUP BY k ORDER BY ABS(MIN(v))"
                .to_string(),
            format!("{groups} HAVING COUNT(*) > (SELECT 0) ORDER BY ABS(MIN(v))"),
            "SELECT DISTINCT ON ((SELECT 1)) w.k FROM w WHERE w.k = t.k ORDER BY ABS(w.v)"
                .to_string(),
        ] {
            let sql = format!("SELECT id FROM t WHERE EXISTS ({block}) ORDER BY id");
            let literal = format!("SELECT id FROM t WHERE EXISTS ({groups}) ORDER BY id");
            holds(&db, &sql, &literal, mode, &mut wrong);
        }
        // Over a join the keys bind with the block, as the join reads it
        let joined = "SELECT u.k FROM u JOIN r ON r.rid = u.id WHERE u.k = t.k GROUP BY u.k";
        for block in [
            format!("{joined} ORDER BY SUM(u.v)"),
            format!("{joined} HAVING COUNT(*) > (SELECT 0) ORDER BY SUM(u.v)"),
        ] {
            let sql = format!("SELECT id FROM t WHERE EXISTS ({block}) ORDER BY id");
            let literal = format!("SELECT id FROM t WHERE EXISTS ({joined}) ORDER BY id");
            holds(&db, &sql, &literal, mode, &mut wrong);
        }
        // Its keys are still bound, as the same ORDER BY is anywhere else
        for (block, error) in [
            (
                "SELECT 1 FROM u WHERE u.k = t.k ORDER BY nosuch LIMIT 1",
                "nosuch",
            ),
            (
                "SELECT k FROM w WHERE w.k = t.k GROUP BY k ORDER BY nosuch",
                "nosuch",
            ),
            (
                "SELECT k FROM w WHERE w.k = t.k GROUP BY k ORDER BY v",
                "Column 'v' not found",
            ),
            (
                "SELECT k, (SELECT 1) FROM w WHERE w.k = t.k GROUP BY k ORDER BY v",
                "Column 'v' not found",
            ),
            (
                "SELECT k, (SELECT 1) FROM w WHERE w.k = t.k GROUP BY k ORDER BY nofn(k)",
                "Function not found: NOFN",
            ),
            (
                "SELECT k FROM w WHERE w.k = t.k GROUP BY k \
                 HAVING COUNT(*) > (SELECT 0) ORDER BY nofn(k)",
                "Function not found: NOFN",
            ),
            (
                "SELECT u.k FROM u JOIN r ON r.rid = u.id WHERE u.k = t.k GROUP BY u.k ORDER BY u.v",
                "Column 'u.v' not found",
            ),
        ] {
            let sql = format!("SELECT id FROM t WHERE EXISTS ({block})");
            if !rows(&db, &sql).is_err_and(|e| e.contains(error)) {
                wrong.push(format!(
                    "{mode}: {sql} binds its key: {:?}",
                    rows(&db, &sql)
                ));
            }
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
