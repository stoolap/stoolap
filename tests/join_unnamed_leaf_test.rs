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

//! A join of more than two sources, one of them a derived table or VALUES
//! list without a name, runs on its prepared blocks: it compiles nothing
//! through the evaluator bridge, names that source's columns bare, and
//! reads a bare name two of its sources have as ambiguous

#![cfg(feature = "test-failpoints")]

use std::sync::Mutex;

use stoolap::test_failpoints::bridge_compiles;
use stoolap::Database;

// The bridge counter is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

/// The columns and rows, sorted, a statement answers, or a text its error
/// holds
type Written = Result<(&'static [&'static str], &'static [&'static str]), &'static str>;

const CASES: &[(&str, Written)] = &[
    (
        "SELECT t.v + 1 FROM (SELECT 1 AS k) JOIN t ON k = t.id JOIN u ON u.tid = t.id",
        Ok((&["t.v + 1"], &["Integer(11)", "Integer(11)"])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS k) JOIN t ON k = t.id JOIN (SELECT 1 AS m) d ON d.m = t.id WHERE d.m > 0",
        Ok((&["k", "t.id", "t.a", "t.v", "d.m"], &["Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(1)"])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS k) JOIN t ON k = t.id JOIN u ON u.tid = t.id WHERE 1 = 0",
        Ok((&["k", "t.id", "t.a", "t.v", "u.id", "u.tid", "u.name"], &[])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS k) JOIN t ON k = t.id JOIN u ON u.tid = t.id WHERE u.id IN (1,2,3) AND LENGTH(u.name) > 3",
        Ok((&["k", "t.id", "t.a", "t.v", "u.id", "u.tid", "u.name"], &["Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(1)|Integer(1)|Text(\"alpha\")"])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS k) JOIN t ON k = t.id JOIN u ON u.tid = t.id WHERE LENGTH(u.name) > 3",
        Ok((&["k", "t.id", "t.a", "t.v", "u.id", "u.tid", "u.name"], &["Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(1)|Integer(1)|Text(\"alpha\")"])),
    ),
    (
        "SELECT * FROM a JOIN (SELECT id AS bid FROM b) ON a.id = bid JOIN c ON a.id = c.id AND a.x < c.y",
        Ok((&["a.id", "a.x", "bid", "c.id", "c.y"], &["Integer(1)|Integer(1)|Integer(1)|Integer(1)|Integer(3)"])),
    ),
    (
        "SELECT * FROM a JOIN (SELECT id AS bid FROM b) ON a.id = bid JOIN c ON a.x < c.y",
        Ok((&["a.id", "a.x", "bid", "c.id", "c.y"], &["Integer(1)|Integer(1)|Integer(1)|Integer(1)|Integer(3)", "Integer(1)|Integer(1)|Integer(1)|Integer(2)|Integer(4)"])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS id) JOIN t ON id = t.id JOIN u ON u.tid = t.id",
        Err("column reference 'id' is ambiguous"),
    ),
    (
        "SELECT t.id FROM (SELECT 1 AS id) JOIN t ON t.id = 1 JOIN u ON u.tid = t.id",
        Ok((&["id"], &["Integer(1)", "Integer(1)"])),
    ),
    (
        "SELECT id FROM (SELECT 1 AS id) JOIN t ON t.id = 1 JOIN u ON u.tid = t.id",
        Err("column reference 'id' is ambiguous"),
    ),
    (
        "SELECT * FROM t JOIN (SELECT 2 AS id, 'zz' AS name) ON t.id = 2 JOIN u ON u.tid = t.id",
        Ok((&["t.id", "t.a", "t.v", "id", "name", "u.id", "u.tid", "u.name"], &["Integer(2)|Integer(2)|Integer(20)|Integer(2)|Text(\"zz\")|Integer(3)|Integer(2)|Text(\"gamma\")"])),
    ),
    (
        "SELECT name FROM t JOIN (SELECT 2 AS k, 'zz' AS name) ON t.id = k JOIN u ON u.tid = t.id",
        Err("column reference 'name' is ambiguous"),
    ),
    (
        "SELECT k, m FROM (SELECT 1 AS k) JOIN (SELECT 1 AS m) ON k = m JOIN t ON t.id = k",
        Ok((&["k", "m"], &["Integer(1)|Integer(1)"])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS k) JOIN (SELECT 1 AS k) ON TRUE JOIN t ON t.id = 1",
        Ok((&["k", "k", "t.id", "t.a", "t.v"], &["Integer(1)|Integer(1)|Integer(1)|Integer(1)|Integer(10)"])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS k) LEFT JOIN t ON k = t.id LEFT JOIN u ON u.tid = t.id",
        Ok((&["k", "t.id", "t.a", "t.v", "u.id", "u.tid", "u.name"], &["Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(1)|Integer(1)|Text(\"alpha\")", "Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(2)|Integer(1)|Text(\"bo\")"])),
    ),
    (
        "SELECT * FROM (VALUES (1)) JOIN t ON t.id = 1 JOIN u ON u.tid = t.id",
        Ok((&["column1", "t.id", "t.a", "t.v", "u.id", "u.tid", "u.name"], &["Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(1)|Integer(1)|Text(\"alpha\")", "Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(2)|Integer(1)|Text(\"bo\")"])),
    ),
    (
        "SELECT COUNT(*), SUM(t.v) FROM (SELECT 1 AS k) JOIN t ON k = t.id JOIN u ON u.tid = t.id",
        Ok((&["COUNT(*)", "SUM(t.v)"], &["Integer(2)|Integer(20)"])),
    ),
    (
        "SELECT * FROM (SELECT 1 AS k) JOIN t ON k = t.id JOIN u ON u.tid = t.id WHERE u.id IN (SELECT id FROM u WHERE name > 'a')",
        Ok((&["k", "t.id", "t.a", "t.v", "u.id", "u.tid", "u.name"], &["Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(1)|Integer(1)|Text(\"alpha\")", "Integer(1)|Integer(1)|Integer(1)|Integer(10)|Integer(2)|Integer(1)|Text(\"bo\")"])),
    ),
    // An empty alias names its table, as any other alias does
    (
        "SELECT \"\".* FROM t AS \"\" CROSS JOIN b",
        Ok((&["id", "a", "v"], &["Integer(1)|Integer(1)|Integer(10)", "Integer(1)|Integer(1)|Integer(10)", "Integer(2)|Integer(2)|Integer(20)", "Integer(2)|Integer(2)|Integer(20)"])),
    ),
    (
        "SELECT \"\".v FROM t AS \"\" JOIN u ON u.tid = \"\".id JOIN b ON b.id = u.id",
        Ok((&["v"], &["Integer(10)", "Integer(10)"])),
    ),
    (
        "SELECT * FROM t AS \"\" JOIN u ON u.tid = \"\".id JOIN b ON b.id = u.id",
        Ok((&[".id", ".a", ".v", "u.id", "u.tid", "u.name", "b.id"], &["Integer(1)|Integer(1)|Integer(10)|Integer(1)|Integer(1)|Text(\"alpha\")|Integer(1)", "Integer(1)|Integer(1)|Integer(10)|Integer(2)|Integer(1)|Text(\"bo\")|Integer(2)"])),
    ),
];

/// The column names and the rows sorted, or the error
type Answer = Result<(Vec<String>, Vec<String>), String>;

fn answer(db: &Database, sql: &str) -> Answer {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
    let mut rows = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        rows.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    rows.sort();
    Ok((columns, rows))
}

fn open() -> Database {
    let db = Database::open("memory://join_unnamed_leaf").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, v INTEGER)",
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER, name TEXT)",
        "CREATE INDEX u_tid ON u (tid)",
        "INSERT INTO u VALUES (1, 1, 'alpha'), (2, 1, 'bo'), (3, 2, 'gamma')",
        "CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO a VALUES (1, 1), (2, 5)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY)",
        "INSERT INTO b VALUES (1), (2)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, y INTEGER)",
        "INSERT INTO c VALUES (1, 3), (2, 4)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn a_join_with_an_unnamed_source_answers_as_written_without_the_bridge() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = open();
    let mut wrong = Vec::new();
    for (sql, want) in CASES {
        for run in 0..2 {
            let before = bridge_compiles();
            let got = answer(&db, sql);
            let compiled = bridge_compiles() - before;
            let fits = match (want, &got) {
                (Ok((columns, rows)), Ok((got_columns, got_rows))) => {
                    got_columns == columns && got_rows == rows
                }
                (Err(want), Err(got)) => got.contains(want),
                _ => false,
            };
            if !fits {
                wrong.push(format!("{sql}, run {run}\n  got  {got:?}\n  want {want:?}"));
            }
            if compiled != 0 {
                wrong.push(format!("{sql}, run {run}: {compiled} bridge compiles"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
