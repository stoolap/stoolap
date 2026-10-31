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

//! A join whose side is itself a join reads that side with the WHERE's
//! conditions on it, and answers as recorded

mod oracle;

use std::sync::{Mutex, MutexGuard};

use stoolap::core::Value;
use stoolap::Database;

// The programs run are counted process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 4, 30), (4, NULL, 40), (5, 3, 50), (6, 1, 60)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9), (5, NULL)",
    "CREATE TABLE r (id INTEGER PRIMARY KEY)",
    "INSERT INTO r VALUES (1), (2), (3), (7)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
    "INSERT INTO w VALUES (1, 1, 1), (2, 2, 6), (3, 3, 5), (4, 1, 2)",
    "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER)",
    "INSERT INTO p VALUES (1, 1), (2, 1), (3, 2), (6, 3)",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://join_nested_side_filter_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// FROMs whose left side is a join: `{j1}` and `{j2}` the joins' kinds
const FROMS: &[&str] = &[
    "t {j1} JOIN u ON u.id = t.a {j2} JOIN r ON r.id = t.id",
    "t {j1} JOIN u ON u.id = t.a {j2} JOIN r ON r.id = u.id",
    "t {j1} JOIN u ON u.id = t.a {j2} JOIN r ON r.id = t.id JOIN p ON p.id = t.id",
];

/// Conditions on the joined side, on one of its tables, by qualified and
/// bare names, with NULLs, subqueries and both sources
const WHERES: &[&str] = &[
    "t.b > 20",
    "u.x IS NULL",
    "u.x > 7 AND t.b < 40",
    "b > 20",
    "x IS NULL",
    "COALESCE(u.x, 0) = 0",
    "t.a = u.id OR t.b > 50",
    "t.b > (SELECT MIN(y) * 10 FROM w)",
    "u.id IN (SELECT x FROM w)",
    "t.b + r.id > 25",
    "r.id > 1 AND t.a IS NOT NULL",
    "id > 1",
    "t.id = 1 AND r.id = (SELECT t.b)",
    "t.id = 1 AND r.id = (SELECT b)",
    "t.b < 50 AND r.id IN (SELECT w.x FROM w WHERE w.y = a)",
    "t.b > 20 AND r.id IN (SELECT w.x FROM w WHERE w.y = t.id)",
    "u.x > 7 AND r.id > (SELECT MIN(w.x) FROM w WHERE w.y < t.b)",
];

const KINDS: &[&str] = &["", "LEFT", "RIGHT", "FULL"];

/// The rows of `sql` with `params` as sorted text, or the error
fn answer(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<String>, String> {
    let rows = db.query(sql, params).map_err(|e| e.to_string())?;
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
    out.sort();
    Ok(out)
}

const RECORDED: oracle::Answers = answers!("join_nested_side_filter");

#[test]
fn a_joined_side_answers_its_conditions_as_recorded() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    let mut count = 0;
    for (context, db) in [("hot", hot("matrix")), ("sealed", sealed(&dir))] {
        for from in FROMS {
            for j1 in KINDS {
                for j2 in KINDS {
                    let from = from.replace("{j1}", j1).replace("{j2}", j2);
                    let sqls = WHERES
                        .iter()
                        .map(|w| {
                            (
                                format!("SELECT t.id, u.id, r.id FROM {from} WHERE {w}"),
                                None,
                            )
                        })
                        .chain([(
                            format!("SELECT t.id, u.id, r.id FROM {from} WHERE t.b > $1"),
                            Some(20),
                        )]);
                    for (sql, param) in sqls {
                        count += 1;
                        let params = param.map(Value::Integer).into_iter().collect();
                        let got = answer(&db, &sql, params);
                        let full = match param {
                            Some(p) => format!("{sql} [$1 = {p}]"),
                            None => sql,
                        };
                        wrong.extend(
                            oracle::check_answer(&RECORDED, &full, &got)
                                .map(|why| format!("{context}: {why}")),
                        );
                    }
                }
            }
        }
    }
    assert!(count > 1000, "{count} statements");
    assert!(
        wrong.is_empty(),
        "{} of {count} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_joined_side_reads_the_values_each_run_gives_its_conditions() {
    let _guard = serial();
    let db = hot("runs");
    let mut wrong = Vec::new();
    for from in [
        "t {j1} JOIN u ON u.id = t.a {j2} JOIN r ON r.id = t.id",
        "t {j1} JOIN u ON u.id = t.a {j2} JOIN r USING (id)",
    ] {
        for j1 in KINDS {
            for j2 in KINDS {
                let from = from.replace("{j1}", j1).replace("{j2}", j2);
                for select in ["t.id, u.id, r.id", "t.id, u.id, r.id, (SELECT 1)"] {
                    let sql = format!("SELECT {select} FROM {from} WHERE t.b > (SELECT $1)");
                    let stmt = db.prepare(&sql).unwrap();
                    for b in [20, 40, 0, 20] {
                        let mut got: Vec<String> = stmt
                            .query((b,))
                            .unwrap()
                            .map(|row| row.unwrap().get::<Option<String>>(0).unwrap())
                            .map(|id| id.unwrap_or_else(|| "NULL".into()))
                            .collect();
                        got.sort();
                        let literal = sql.replace("(SELECT $1)", &b.to_string());
                        let mut want: Vec<String> = answer(&db, &literal, Vec::new())
                            .unwrap()
                            .into_iter()
                            .map(|row| row.split('|').next().unwrap_or_default().to_string())
                            .collect();
                        want.sort();
                        if got != want {
                            wrong.push(format!("{sql} [$1 = {b}]: {got:?}, want {want:?}"));
                        }
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_joined_side_reads_only_the_rows_its_conditions_pass() {
    let _guard = serial();
    let db = Database::open("memory://join_nested_side_filter_reads").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t SELECT g.value, g.value FROM generate_series(1, 1000) g",
        "CREATE TABLE u (id INTEGER PRIMARY KEY)",
        "INSERT INTO u SELECT g.value FROM generate_series(1, 1000) g",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "INSERT INTO r SELECT g.value FROM generate_series(1, 1000) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT t.id FROM t JOIN u ON u.id = t.a JOIN r ON r.id = t.id \
               WHERE t.id < 3 AND t.a + 0 > 0";
    let mut rows = 0;
    let ran = stoolap::test_failpoints::programs_ran(|| {
        rows = db.query(sql, ()).unwrap().count();
    });
    let entries: usize = ran.iter().map(|(_, times)| times).sum();
    assert_eq!(rows, 2);
    assert!(
        entries < 100,
        "the joined side's rows were asked {entries} times for 2: {ran:?}"
    );
}
