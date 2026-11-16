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

use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::{test_failpoints, Database};

// The side file counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

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

/// The side file's candidate positions and the volumes it left to the scan
/// as too many, so far
fn side_reads(db: &Database) -> (i64, i64) {
    let rows = db.query("PRAGMA INDEX_READ_STATS", ()).unwrap();
    let columns = rows.columns().to_vec();
    let at = |name: &str| columns.iter().position(|c| c == name).unwrap();
    let (candidates, scans) = (at("candidates"), at("cost_scans"));
    let row = rows.into_iter().next().unwrap().unwrap();
    (
        row.get::<i64>(candidates).unwrap(),
        row.get::<i64>(scans).unwrap(),
    )
}

/// Hot index ids the reads took so far
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
    ]
}

fn check(db: &Database, sealed: bool, wrong: &mut Vec<String>) {
    let at = if sealed { "sealed" } else { "hot" };
    for (sql, params, expected, most) in cases() {
        let (candidates, scans) = side_reads(db);
        let hot = hot_ids();
        let answered = ids(db, &sql, params);
        if answered != expected {
            wrong.push(format!(
                "{at} {sql}: answered {answered:?}, expected {expected:?}"
            ));
        }
        let (candidates, scans) = (side_reads(db).0 - candidates, side_reads(db).1 - scans);
        let hot = hot_ids() - hot;
        let read = if sealed { candidates as usize } else { hot };
        if read > most || scans != 0 {
            wrong.push(format!(
                "{at} {sql}: read {read} ids, {scans} volumes scanned for too many, at most {most}"
            ));
        }
    }
}

#[test]
fn a_read_starts_from_the_index_holding_the_fewest_rows() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
