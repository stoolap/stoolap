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

//! An UPDATE, DELETE or INSERT compiles its programs once, with its plan: a
//! run after the first compiles nothing, hot and sealed, in a transaction or
//! not

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{program_compiles, programs_ran};
use stoolap::{Database, Value};

/// Held by each test: the programs one records are those of every thread
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Each statement with the parameters of its first and its second run
const CASES: &[(&str, [&[i64]; 2])] = &[
    ("UPDATE t SET b = b + $1 WHERE id = $2", [&[1, 5], &[1, 6]]),
    ("UPDATE t SET b = b * 2 WHERE a + b > $1", [&[100], &[200]]),
    (
        "UPDATE t SET b = 1 WHERE id = 7 RETURNING id, b * $1",
        [&[2], &[3]],
    ),
    ("UPDATE t SET b = 2 WHERE id = 8 RETURNING *", [&[], &[]]),
    ("DELETE FROM t WHERE id = $1", [&[40], &[41]]),
    ("DELETE FROM t WHERE a * 2 = $1", [&[18], &[16]]),
    (
        "DELETE FROM t WHERE id = $1 RETURNING id, a + 1",
        [&[42], &[43]],
    ),
    ("DELETE FROM t WHERE id = $1 RETURNING *", [&[44], &[45]]),
    (
        "INSERT INTO t VALUES ($1, $2 * 2, ABS(-3), 1.0, 'n' || 'x', NULL, $1, 1)",
        [&[101, 1], &[102, 2]],
    ),
    (
        "INSERT INTO t (id, a, u, c) VALUES ($1, 1 + 1, $1, 1), ($1 + 1, 1 + 1, $1 + 1, 1)",
        [&[103], &[105]],
    ),
    (
        "INSERT INTO t (id, u, c) VALUES ($1, $1, 1) RETURNING id, a, c + 1",
        [&[107], &[108]],
    ),
    (
        "INSERT INTO t VALUES ($1, 1, 1, 1.0, 'n', 1, $1 + 1000, 1) \
         ON CONFLICT (id) DO UPDATE SET b = b + 1",
        [&[3], &[109]],
    ),
    ("UPDATE k SET v = v + 1 WHERE id = $1", [&[1], &[2]]),
    ("INSERT INTO k (id, v) VALUES ($1, $1 * 2)", [&[10], &[11]]),
    (
        "INSERT INTO k (id, v) VALUES ($1, 2) ON CONFLICT (id) DO UPDATE SET v = v + 1",
        [&[1], &[12]],
    ),
];

/// t with 50 rows and k with CHECK and a computed default; sealed when `cold`
fn tables(name: &str, cold: bool, dir: &tempfile::TempDir) -> Database {
    let db = match cold {
        true => Database::open(&format!(
            "file://{}?checkpoint_interval=3600&compact_threshold=1000",
            dir.path().display()
        )),
        false => Database::open(&format!("memory://{name}")),
    }
    .unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, f FLOAT, s TEXT, \
         n INTEGER, u INTEGER UNIQUE, c INTEGER NOT NULL)",
        "INSERT INTO t SELECT value, value % 10, value * 3, value * 0.75, 'x' || value, \
         value, value, 1 FROM generate_series(1, 50)",
        "CREATE TABLE k (id INTEGER PRIMARY KEY, v INTEGER CHECK (v > 0), \
         w INTEGER DEFAULT (ABS(-5)))",
        "INSERT INTO k (id, v) VALUES (1, 1), (2, 2), (3, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Run `sql` once with `params`, its RETURNING rows read
fn run(db: &Database, sql: &str, params: &[i64], in_transaction: bool) -> Result<(), String> {
    let params: Vec<Value> = params.iter().map(|v| Value::Integer(*v)).collect();
    if in_transaction {
        db.execute("BEGIN", ()).unwrap();
    }
    let ran = if sql.contains("RETURNING") {
        db.query(sql, params)
            .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>().map(|_| ()))
    } else {
        db.execute(sql, params).map(|_| ())
    };
    if in_transaction {
        db.execute("COMMIT", ()).unwrap();
    }
    ran.map_err(|e| format!("{sql}: {e}"))
}

#[test]
fn a_dml_run_after_the_first_compiles_nothing() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut wrong = Vec::new();
    for (cold, in_transaction) in [(false, false), (true, false), (false, true), (true, true)] {
        let dir = tempfile::tempdir().unwrap();
        let db = tables(
            &format!("dml_held_programs_{cold}_{in_transaction}"),
            cold,
            &dir,
        );
        for &(sql, [first, second]) in CASES {
            if let Err(e) = run(&db, sql, first, in_transaction) {
                wrong.push(e);
                continue;
            }
            let programs = program_compiles();
            let again = run(&db, sql, second, in_transaction);
            let compiled = program_compiles() - programs;
            if again.is_err() || compiled != 0 {
                wrong.push(format!(
                    "{sql} (sealed {cold}, in a transaction {in_transaction}): \
                     {again:?}, run again, compiled {compiled:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Each RETURNING reading a parameter, over the rows it returns
const RETURNING_PARAMETERS: &[&str] = &[
    "INSERT INTO dst SELECT id, v FROM src RETURNING id, VEC_DISTANCE_L2(v, $1)",
    "UPDATE src SET n = n + 1 RETURNING VEC_DISTANCE_L2(v, $1)",
    "DELETE FROM dst RETURNING id, VEC_DISTANCE_L2(v, $1)",
];

#[test]
fn a_returning_binds_its_parameters_once_per_run() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://dml_held_returning_parameters").unwrap();
    for sql in [
        "CREATE TABLE src (id INTEGER PRIMARY KEY, v VECTOR(3), n INTEGER)",
        "CREATE TABLE dst (id INTEGER PRIMARY KEY, v VECTOR(3))",
        "INSERT INTO src SELECT value, '[1,2,3]', 0 FROM generate_series(1, 5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for sql in RETURNING_PARAMETERS {
        let mut returned = 0;
        let ran = programs_ran(|| {
            returned = db
                .query(sql, vec![Value::text("[1,2,4]")])
                .and_then(|rows| rows.collect::<stoolap::Result<Vec<_>>>())
                .unwrap_or_else(|e| panic!("{sql}: {e}"))
                .len();
        });
        let per_row: Vec<String> = ran
            .into_iter()
            .filter(|(listing, times)| *times > 1 && listing.contains("LoadParam"))
            .map(|(listing, times)| format!("{times} runs of {listing}"))
            .collect();
        if returned != 5 || !per_row.is_empty() {
            wrong.push(format!("{sql}: {returned} rows, {per_row:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
