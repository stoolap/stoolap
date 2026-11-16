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

//! Where several indexes answer terms of one WHERE, the read starts from the
//! index whose key holds the fewest rows, whatever order the terms are
//! written in, hot and sealed

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::{test_failpoints, Database};

const ROWS: i64 = 4000;

/// The `b` of row `id`: every row a value of its own
fn b(id: i64) -> i64 {
    id * 7919 % ROWS
}

fn open(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER, a INTEGER, b INTEGER, s INTEGER, pad TEXT)"
            .to_string(),
        format!(
            "INSERT INTO t SELECT g.value, g.value, g.value % 100, g.value % 2, g.value * 7919 % {ROWS}, \
             CASE WHEN g.value % 1000 = 7 THEN 1 ELSE 0 END, 'p' FROM generate_series(1, {ROWS}) g"
        ),
        // a's index is created first, the one a rule by identity would take
        "CREATE INDEX t_a ON t (a)".to_string(),
        "CREATE INDEX t_b ON t (b)".to_string(),
        "CREATE INDEX t_w ON t (w)".to_string(),
        "CREATE INDEX t_v ON t (v)".to_string(),
        "CREATE INDEX t_s ON t (s)".to_string(),
        // b holds two values far apart, so a side file directory's estimate
        // of b = 0 misjudges it; c two text values, d one a row
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c TEXT, d TEXT)".to_string(),
        format!(
            "INSERT INTO x SELECT g.value, g.value % 100, \
             CASE WHEN g.value % 2 = 0 THEN 0 ELSE 1000000000 END, \
             'c' || (g.value % 2), 'd' || g.value FROM generate_series(1, {ROWS}) g"
        ),
        "CREATE INDEX x_a ON x (a)".to_string(),
        "CREATE INDEX x_b ON x (b)".to_string(),
        "CREATE INDEX x_c ON x (c)".to_string(),
        "CREATE INDEX x_d ON x (d)".to_string(),
    ] {
        db.execute(&sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn ids(db: &Database, sql: &str, params: Vec<Value>) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// Hot index ids this thread's reads took so far
fn hot_ids() -> usize {
    test_failpoints::hot_index_ids_read() + test_failpoints::equality_candidate_ids_copied_count()
}

/// Each statement, its parameters, its rows, and how many ids or candidate
/// positions the index reads may take for it at most
fn cases() -> Vec<(String, Vec<Value>, Vec<i64>, usize)> {
    let key = b(1001);
    let odd_37: Vec<i64> = (0..ROWS / 100).map(|n| n * 100 + 37).collect();
    vec![
        (
            format!("SELECT id FROM t WHERE a = 1 AND b = {key}"),
            vec![],
            vec![1001],
            8,
        ),
        (
            format!("SELECT id FROM t WHERE b = {key} AND a = 1"),
            vec![],
            vec![1001],
            8,
        ),
        (
            format!("SELECT id FROM t WHERE a = 1 AND b = {key} + 0"),
            vec![],
            vec![1001],
            8,
        ),
        (
            format!("SELECT id FROM t WHERE a = 0 AND b = {key}"),
            vec![],
            vec![],
            8,
        ),
        (
            "SELECT id FROM t WHERE a = $1 AND b = $2".to_string(),
            vec![Value::Integer(1), Value::Integer(key)],
            vec![1001],
            8,
        ),
        (
            "SELECT id FROM t WHERE b = $2 AND a = $1".to_string(),
            vec![Value::Integer(1), Value::Integer(key)],
            vec![1001],
            8,
        ),
        (
            format!("SELECT id FROM t WHERE a = 1 AND w = 1 AND b = {key}"),
            vec![],
            vec![1001],
            8,
        ),
        (
            "SELECT id FROM t WHERE a = 1 AND w = 37".to_string(),
            vec![],
            odd_37,
            64,
        ),
        (
            format!("SELECT id FROM t WHERE a = 1 AND v > {}", ROWS - 10),
            vec![],
            vec![3991, 3993, 3995, 3997, 3999],
            16,
        ),
        (
            format!("SELECT id FROM t WHERE s = 0 AND b = {key}"),
            vec![],
            vec![1001],
            8,
        ),
        (
            format!(
                "SELECT id FROM t WHERE a = 1 AND b IN ({key}, {}, {})",
                b(1002),
                b(1003)
            ),
            vec![],
            vec![1001, 1003],
            8,
        ),
        (
            "SELECT id FROM x WHERE a = 2 AND b = 0".to_string(),
            vec![],
            (0..ROWS / 100).map(|n| n * 100 + 2).collect(),
            64,
        ),
        (
            "SELECT id FROM x WHERE c = 'c1' AND d = 'd1001'".to_string(),
            vec![],
            vec![1001],
            8,
        ),
    ]
}

fn check(db: &Database, sealed: bool, wrong: &mut Vec<String>) {
    let at = if sealed { "sealed" } else { "hot" };
    for (sql, params, expected, most) in cases() {
        let (candidates, scans) = test_failpoints::side_decisions();
        let hot = hot_ids();
        let answered = ids(db, &sql, params);
        if answered != expected {
            wrong.push(format!(
                "{at} {sql}: answered {answered:?}, expected {expected:?}"
            ));
        }
        let after = test_failpoints::side_decisions();
        let (candidates, scans) = (after.0 - candidates, after.1 - scans);
        let hot = hot_ids() - hot;
        // Every key named exists, so an index read takes one id at least; a
        // volume narrows a range on its sorted column and an IN list itself,
        // with no side file walk
        let read = if sealed { candidates } else { hot };
        let walked = !sealed || !(sql.contains(" v > ") || sql.contains(" IN ("));
        if (walked && read == 0) || read > most || scans != 0 {
            wrong.push(format!(
                "{at} {sql}: read {read} ids, {scans} volumes scanned for too many, \
                 1 to {most} expected"
            ));
        }
    }
}

#[test]
fn a_read_starts_from_the_index_holding_the_fewest_rows() {
    let mut wrong = Vec::new();
    let db = open("memory://index_fewest_rows_hot", false);
    check(&db, false, &mut wrong);
    let dir = tempfile::tempdir().unwrap();
    let db = open(
        &format!("file://{}?checkpoint_interval=3600", dir.path().display()),
        true,
    );
    check(&db, true, &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn local_writes_and_dml_read_the_fewest_rows_index_too() {
    let key = b(1001);
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = match sealed {
            true => format!("file://{}?checkpoint_interval=3600", dir.path().display()),
            false => "memory://index_fewest_rows_writes".to_string(),
        };
        let db = open(&dsn, sealed);
        let sql = format!("SELECT id FROM t WHERE a = 1 AND b = {key}");
        db.execute("BEGIN", ()).unwrap();
        db.execute(
            &format!("INSERT INTO t VALUES (5001, 5001, 1, 1, {key}, 0, 'q')"),
            (),
        )
        .unwrap();
        assert_eq!(ids(&db, &sql, vec![]), [1001, 5001], "a local row");
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(ids(&db, &sql, vec![]), [1001], "rolled back");
        let updated = db
            .execute(
                &format!("UPDATE t SET pad = 'z' WHERE a = 1 AND b = {key}"),
                (),
            )
            .unwrap();
        assert_eq!(updated, 1);
        assert_eq!(
            ids(&db, "SELECT id FROM t WHERE pad = 'z'", vec![]),
            [1001],
            "the updated row"
        );
        let deleted = db
            .execute(&format!("DELETE FROM t WHERE b = {key} AND a = 1"), ())
            .unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(ids(&db, &sql, vec![]), Vec::<i64>::new(), "deleted");
    }
}

#[test]
fn explain_names_the_fewest_rows_choice() {
    let db = open("memory://index_fewest_rows_explain", false);
    let plan = |sql: &str| -> String {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(0).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let key = b(1001);
    let keyed = plan(&format!("SELECT id FROM t WHERE a = 1 AND b = {key}"));
    assert!(
        keyed.contains(&format!("IndexGet t by the fewest of a = 1, b = {key}"))
            && keyed.contains("Intersect, fewest rows first"),
        "{keyed}"
    );
    let single = plan("SELECT id FROM t WHERE w = 37");
    assert!(
        single.contains("IndexGet t w = 37") && !single.contains("fewest"),
        "{single}"
    );
}

#[test]
fn a_range_below_a_value_reads_and_counts_no_null_key() {
    let db = Database::open("memory://index_fewest_rows_nulls").unwrap();
    for sql in [
        "CREATE TABLE n (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
        "INSERT INTO n SELECT g.value, \
         CASE WHEN g.value <= 3500 THEN NULL WHEN g.value <= 3600 THEN 1 ELSE 9 END, \
         CASE WHEN g.value > 3400 THEN 1 ELSE 0 END FROM generate_series(1, 4000) g",
        "CREATE INDEX n_x ON n (x)",
        "CREATE INDEX n_y ON n (y)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for sql in [
        "SELECT id FROM n WHERE x < 5 AND y = 1",
        "SELECT id FROM n WHERE x <= 1 AND y = 1",
    ] {
        let before = hot_ids();
        assert_eq!(
            ids(&db, sql, vec![]),
            (3501..=3600).collect::<Vec<_>>(),
            "{sql}"
        );
        let read = hot_ids() - before;
        assert!(
            read <= 600,
            "{sql}: x's 100 rows below 5 or the 600 of y = 1 read, not the 3,500 NULLs: {read}"
        );
    }
}
