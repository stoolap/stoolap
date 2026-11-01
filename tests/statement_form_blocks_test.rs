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
//! statement reads, run prepared as statements.

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
fn an_exists_statement_block_without_an_order_by_runs_prepared() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        let mode = if sealed { "sealed" } else { "hot" };
        let groups = "SELECT k FROM w WHERE w.k = t.k GROUP BY k";
        let none = "SELECT 1 FROM w WHERE 1 = 0";
        for (block, twin) in [
            (format!("{groups} LIMIT 1"), groups),
            (format!("{groups} LIMIT 0"), none),
            (format!("{groups} OFFSET 1"), none),
            // Its SELECT list is never computed, so this subquery of two rows raises nothing
            (
                "SELECT k, (SELECT v FROM u) FROM w WHERE w.k = t.k GROUP BY k".to_string(),
                groups,
            ),
            (
                format!("{groups} HAVING COUNT(*) > (SELECT 1)"),
                "SELECT k FROM w WHERE w.k = t.k GROUP BY k HAVING COUNT(*) > 1",
            ),
        ] {
            let sql = format!("SELECT id FROM t WHERE EXISTS ({block}) ORDER BY id");
            let literal = format!("SELECT id FROM t WHERE EXISTS ({twin}) ORDER BY id");
            holds(&db, &sql, &literal, mode, &mut wrong);
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn an_exists_ordered_or_distinct_where_only_a_statement_reads_it_computes_neither_keys_nor_list() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup(&dir, sealed);
        let mode = if sealed { "sealed" } else { "hot" };
        let groups = "SELECT k FROM w WHERE w.k = t.k GROUP BY k";
        let joined = "SELECT w.k FROM w JOIN r ON r.rid = w.id WHERE w.k = t.k GROUP BY w.k";
        let rows_of = "SELECT 1 FROM w WHERE w.k = t.k";
        let none = "SELECT 1 FROM w WHERE 1 = 0";
        let big = "9223372036854775807";
        for (block, twin) in [
            (format!("{groups} ORDER BY SUM(v)"), groups.to_string()),
            (format!("{groups} ORDER BY k LIMIT 1"), groups.to_string()),
            (
                "SELECT k, (SELECT 1) FROM w WHERE w.k = t.k GROUP BY k ORDER BY k".to_string(),
                groups.to_string(),
            ),
            (
                format!("{groups} HAVING COUNT(*) > (SELECT 0) ORDER BY k"),
                format!("{groups} HAVING COUNT(*) > 0"),
            ),
            (
                format!("{rows_of} ORDER BY (SELECT 1) LIMIT 1"),
                rows_of.to_string(),
            ),
            (format!("{joined} ORDER BY SUM(w.v)"), joined.to_string()),
            (
                format!("{groups} ORDER BY SUM(v) * {big} LIMIT 1"),
                groups.to_string(),
            ),
            (
                format!("{groups} ORDER BY (SELECT MAX(v) FROM u) - MAX(v) LIMIT 1"),
                groups.to_string(),
            ),
            (
                "SELECT k AS kk FROM w WHERE w.k = t.k GROUP BY k ORDER BY kk, COUNT(*) LIMIT 1"
                    .to_string(),
                groups.to_string(),
            ),
            (
                "SELECT ABS(MIN(v)) FROM w WHERE w.k = t.k GROUP BY k ORDER BY k LIMIT 1".to_string(),
                groups.to_string(),
            ),
            (
                format!("SELECT ABS(v) FROM w WHERE w.k = t.k ORDER BY v * {big} LIMIT (SELECT 1)"),
                rows_of.to_string(),
            ),
            (
                "SELECT DISTINCT ON ((SELECT 1)) ABS(v) FROM w WHERE w.k = t.k".to_string(),
                rows_of.to_string(),
            ),
            (
                "SELECT DISTINCT ON ((SELECT 1)) k FROM w WHERE w.k = t.k OFFSET 1".to_string(),
                none.to_string(),
            ),
            (
                format!("SELECT DISTINCT ON (1) k, ABS(v) FROM w WHERE w.k = t.k ORDER BY v * {big} OFFSET 1"),
                none.to_string(),
            ),
            (
                "SELECT DISTINCT ON (kk) k AS kk FROM w WHERE w.k = t.k OFFSET (SELECT 1)".to_string(),
                none.to_string(),
            ),
            (
                "SELECT DISTINCT ON (MAX(v)) k FROM w WHERE w.k = t.k GROUP BY k OFFSET 1".to_string(),
                none.to_string(),
            ),
            (
                "SELECT DISTINCT ON (2) * FROM w WHERE w.k = t.k OFFSET 1".to_string(),
                none.to_string(),
            ),
        ] {
            for site in [
                "SELECT id FROM t WHERE EXISTS (?) ORDER BY id",
                "SELECT id FROM t WHERE NOT EXISTS (?) ORDER BY id",
                "SELECT id, EXISTS (?) FROM t ORDER BY id",
                "SELECT id, CASE WHEN EXISTS (?) THEN 'y' ELSE 'n' END FROM t ORDER BY id",
            ] {
                let (sql, twin) = (site.replace('?', &block), site.replace('?', &twin));
                holds(&db, &sql, &twin, mode, &mut wrong);
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn an_exists_ordered_where_only_a_statement_reads_it_reads_a_parameter_prepared() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup(&dir, false);
    let sql = "SELECT id FROM t WHERE EXISTS (SELECT k FROM w WHERE w.k = t.k AND w.v > $1 \
               GROUP BY k ORDER BY SUM(v) * 9223372036854775807 LIMIT $2) ORDER BY id";
    let statement = db.prepare(sql).unwrap();
    let mut got = Vec::new();
    for (floor, limit) in [(0, 1), (4, 1), (0, 0)] {
        let ids: Vec<i64> = statement
            .query((floor, limit))
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        got.push(ids);
    }
    assert_eq!(got, vec![vec![1, 2], vec![2], vec![]]);
    assert_eq!(prepared_line(&db, sql), "Prepared: yes");
}

#[test]
fn an_exists_ordered_or_distinct_binds_what_it_does_not_compute() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup(&dir, false);
    let mut wrong = Vec::new();
    for (block, error) in [
        (
            "SELECT k FROM w WHERE w.k = t.k AND 1 = 0 GROUP BY k ORDER BY nocol LIMIT 1",
            "nocol",
        ),
        (
            "SELECT k FROM w WHERE w.k = t.k GROUP BY k ORDER BY v LIMIT 1",
            "Column 'v' not found",
        ),
        (
            "SELECT k FROM w WHERE w.k = t.k GROUP BY k ORDER BY NOFUNC(k) LIMIT 1",
            "NOFUNC",
        ),
        (
            "SELECT COUNT(*) FROM w WHERE w.k = t.k ORDER BY v LIMIT 1",
            "Column 'v' not found",
        ),
        (
            "SELECT v FROM w WHERE w.k = t.k ORDER BY (SELECT MAX(nocol) FROM u)",
            "nocol",
        ),
        (
            "SELECT DISTINCT ON (3) k FROM w WHERE w.k = t.k OFFSET 1",
            "DISTINCT ON key 3 names no column",
        ),
    ] {
        let sql = format!("SELECT id FROM t WHERE EXISTS ({block}) ORDER BY id");
        match rows(&db, &sql) {
            Err(e) if e.contains(error) => {}
            other => wrong.push(format!("{sql}: {other:?}, want an error naming {error}")),
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
