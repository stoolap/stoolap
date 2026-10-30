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

//! Texts that differ only in the literals a WHERE compares with a column
//! share one plan, and answer, name their columns and fail as each text
//! would under a plan of its own

#![cfg(feature = "test-failpoints")]

use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    shared_literals, shared_plan_runs, SharedLiteral, LITERAL_SHARING_OFF,
};
use stoolap::Database;

/// The switch is process-wide: one test reads with it at a time, and puts
/// it back on
static SWITCH: Mutex<()> = Mutex::new(());

struct Sharing<'a> {
    _serial: MutexGuard<'a, ()>,
}

impl Sharing<'_> {
    fn set(on: bool) -> Self {
        let serial = SWITCH.lock().unwrap_or_else(|p| p.into_inner());
        LITERAL_SHARING_OFF.store(!on, Ordering::Relaxed);
        Self { _serial: serial }
    }
}

impl Drop for Sharing<'_> {
    fn drop(&mut self) {
        LITERAL_SHARING_OFF.store(false, Ordering::Relaxed);
    }
}

/// Every statement of a run, in order; a pair of texts of one skeleton
/// follow each other so the second reads the plan the first made
const RUN: &[&str] = &[
    "SELECT id, name FROM u WHERE id = 3",
    "SELECT id, name FROM u WHERE id = 4",
    "SELECT id FROM u WHERE 21 = code",
    "SELECT id FROM u WHERE 28 = code",
    "SELECT id FROM u WHERE name = 'n5' OR name = 'n6' ORDER BY id",
    "SELECT id FROM u WHERE name = 'n7' OR name = 'n8' ORDER BY id",
    "SELECT id FROM u WHERE NOT (age = 19) AND id < 6 ORDER BY id",
    "SELECT id FROM u WHERE NOT (age = 20) AND id < 7 ORDER BY id",
    "SELECT id FROM u WHERE v >= 3.5 AND v <= 5.5 ORDER BY id",
    "SELECT id FROM u WHERE v > 3 AND v < 6 ORDER BY id",
    "SELECT id FROM u WHERE v > 3.0 AND v < 6.0 ORDER BY id",
    "SELECT id FROM u WHERE id <> 2 AND id <= 3 ORDER BY id",
    "SELECT id FROM u WHERE id BETWEEN 2 AND 4 ORDER BY id",
    "SELECT id FROM u WHERE id BETWEEN 5 AND 6 ORDER BY id",
    "SELECT id FROM u WHERE id NOT BETWEEN 3 AND 19 ORDER BY id",
    "SELECT id FROM u WHERE id IN (1, 2, 3) ORDER BY id",
    "SELECT id FROM u WHERE id IN (7, 8, 9) ORDER BY id",
    "SELECT id FROM u WHERE id NOT IN (1, 2, 3) AND id < 6 ORDER BY id",
    "SELECT id FROM u WHERE name IN ('n1', 'n9') ORDER BY id",
    "SELECT id FROM u WHERE name IN ('n2', 'n3') ORDER BY id",
    "SELECT id FROM u WHERE id = -0",
    "SELECT id FROM u WHERE id = -1",
    "SELECT id FROM u WHERE v = -0.0",
    "SELECT id FROM u WHERE v = -1.5",
    "SELECT id FROM u WHERE id = 1 AND id = $18446744073709551615",
    "SELECT id FROM u WHERE id = -5",
    "SELECT id FROM u WHERE id = 5",
    "SELECT id FROM u WHERE id = 99999999999999999999",
    "SELECT id FROM u WHERE id = 6",
    "SELECT id FROM u WHERE id = 6.0",
    "SELECT id FROM u WHERE id = 6.5",
    "SELECT id FROM u WHERE name = '{\"a\": 1}'",
    "SELECT id FROM u WHERE name = 'n7'",
    "SELECT id FROM u WHERE name = 'it''s'",
    "SELECT id FROM u WHERE code = 'x'",
    "SELECT id FROM u WHERE code = '14'",
    "SELECT id FROM u WHERE name > 5 ORDER BY id",
    "SELECT 1 AS one, id FROM u WHERE id = 2",
    "SELECT 2 AS one, id FROM u WHERE id = 3",
    "SELECT id, v + 1 FROM u WHERE id = 2",
    "SELECT id, v + 2 FROM u WHERE id = 3",
    "SELECT id FROM u WHERE id > 1 ORDER BY 1 DESC LIMIT 2",
    "SELECT id FROM u WHERE id > 2 ORDER BY 1 LIMIT 3",
    "SELECT id FROM u WHERE id > 15 ORDER BY id LIMIT 2 OFFSET 1",
    "SELECT id FROM u WHERE id = 3 -- tail",
    "SELECT id FROM u WHERE id = 4 -- tail",
    "SELECT id FROM u WHERE id = 5 /* open",
    "select id from u where id = 6",
    "SELECT id FROM u WHERE id = 1 + 1",
    "SELECT id FROM u WHERE id = 2 + 1",
    "SELECT id FROM u WHERE ABS(id) = 3",
    "SELECT id FROM u WHERE id = 3 AND name LIKE 'n%'",
    "SELECT id FROM u WHERE id = 4 AND name LIKE 'x%'",
    "SELECT * FROM u WHERE age = 19 ORDER BY id",
    "SELECT * FROM u WHERE age = 20",
    "SELECT * FROM u WHERE age = 19",
    "SELECT * FROM u WHERE id > 17",
    "SELECT * FROM u WHERE id > 18",
    "SELECT COUNT(*) FROM u WHERE age = 19",
    "SELECT COUNT(*) FROM u WHERE age = 20",
    "SELECT age, COUNT(*) FROM u WHERE id < 10 GROUP BY age ORDER BY age",
    "SELECT age, COUNT(*) FROM u WHERE id < 12 GROUP BY age ORDER BY age",
    "UPDATE u SET v = 1.25 WHERE id = 3",
    "UPDATE u SET v = 2.25 WHERE id = 4",
    "UPDATE u SET v = v + 1 WHERE age = 21",
    "UPDATE u SET v = v + 1 WHERE age = 22",
    "UPDATE u SET code = 700 WHERE id = 9",
    "DELETE FROM u WHERE id = 20",
    "DELETE FROM u WHERE id = 19",
    "DELETE FROM u WHERE id IN (17, 18)",
    "SELECT id, code, name, v, age FROM u ORDER BY id",
];

/// What a statement answers: its columns and its rows with their types, or
/// the rows it changed, or its error
fn answer(db: &Database, sql: &str, params: Option<(i64,)>) -> String {
    if !sql.starts_with("SELECT") {
        return match db.execute(sql, ()) {
            Ok(changed) => format!("changed {changed}"),
            Err(error) => format!("error {error}"),
        };
    }
    let rows = match params {
        Some(params) => db.query(sql, params),
        None => db.query(sql, ()),
    };
    match rows {
        Ok(rows) => {
            let columns = rows.columns().join(", ");
            let rows: Vec<String> = rows
                .map(|row| match row {
                    Ok(row) => format!("{:?}", row.as_row()),
                    Err(error) => format!("error {error}"),
                })
                .collect();
            format!("[{columns}] {rows:?}")
        }
        Err(error) => format!("error {error}"),
    }
}

/// Rows 1 to 20, code 7 * id; sealed into a volume when `cold`
fn table(name: &str, cold: bool, dir: &tempfile::TempDir) -> Database {
    let db = match cold {
        true => Database::open(&format!(
            "file://{}?checkpoint_interval=3600&compact_threshold=1000",
            dir.path().display()
        )),
        false => Database::open(&format!("memory://{name}")),
    }
    .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, name TEXT, \
         v FLOAT, age INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX u_age ON u(age)", ()).unwrap();
    for id in 1..=20i64 {
        db.execute(
            "INSERT INTO u VALUES ($1, $2, $3, $4, $5)",
            (id, id * 7, format!("n{id}"), id as f64 + 0.5, 18 + id % 5),
        )
        .unwrap();
    }
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The answers of `RUN` with sharing `on`, on a table hot or sealed, in a
/// transaction or not
fn answers(name: &str, on: bool, cold: bool, in_transaction: bool) -> Vec<String> {
    let _sharing = Sharing::set(on);
    let dir = tempfile::tempdir().unwrap();
    let db = table(name, cold, &dir);
    if in_transaction {
        db.execute("BEGIN", ()).unwrap();
    }
    let mut answers: Vec<String> = RUN.iter().map(|sql| answer(&db, sql, None)).collect();
    // The statement's own parameter comes before the shared literals
    for (sql, id) in [
        ("SELECT id FROM u WHERE id = $1 AND age = 19", 11),
        ("SELECT id FROM u WHERE id = $1 AND age = 21", 13),
        ("SELECT id FROM u WHERE id = $1 AND age = 21", 3),
        ("SELECT id FROM u WHERE age = 21", 5),
    ] {
        answers.push(answer(&db, sql, Some((id,))));
    }
    for v in [7.5, 8.5] {
        let changed = db.execute("UPDATE u SET v = $1 WHERE id = 2", (v,));
        answers.push(format!("{changed:?}"));
    }
    answers.push(answer(&db, "SELECT v FROM u WHERE id = 2", None));
    if in_transaction {
        db.execute("COMMIT", ()).unwrap();
    }
    answers
}

fn compare(name: &str, cold: bool, in_transaction: bool) {
    let own = answers(&format!("{name}_own"), false, cold, in_transaction);
    let shared = answers(&format!("{name}_shared"), true, cold, in_transaction);
    let statements = RUN.iter().copied().chain([
        "param 11 age 19",
        "param 13 age 21",
        "param 3 age 21",
        "extra param age 21",
        "set v 7.5",
        "set v 8.5",
        "v of 2",
    ]);
    let differ: Vec<String> = statements
        .zip(own.iter().zip(&shared))
        .filter(|(_, (own, shared))| own != shared)
        .map(|(sql, (own, shared))| format!("{sql}\n  own:    {own}\n  shared: {shared}"))
        .collect();
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

#[test]
fn shared_plans_answer_as_plans_of_their_own_hot_and_sealed() {
    let before = SharedLiteral::ALL.map(shared_literals);
    let runs = shared_plan_runs();
    for (cold, in_transaction) in [(false, false), (true, false), (false, true), (true, true)] {
        compare(
            &format!("shared_literals_{cold}_{in_transaction}"),
            cold,
            in_transaction,
        );
    }
    let after = SharedLiteral::ALL.map(shared_literals);
    for (place, (before, after)) in SharedLiteral::ALL.iter().zip(before.iter().zip(after)) {
        assert!(after > *before, "no case shares a literal at {place:?}");
    }
    assert!(shared_plan_runs() > runs, "no text ran a shared plan");
}

#[test]
fn a_text_differing_in_its_shared_literals_runs_the_plan_the_first_made() {
    let _sharing = Sharing::set(true);
    let db = Database::open("memory://shared_literals_reuse").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
        .unwrap();
    let first: i64 = db
        .query_one("SELECT a FROM t WHERE id = 1 AND a > 5", ())
        .unwrap();
    let runs = shared_plan_runs();
    let second: i64 = db
        .query_one("SELECT a FROM t WHERE id = 2 AND a > 15", ())
        .unwrap();
    assert_eq!((first, second), (10, 20));
    assert_eq!(
        shared_plan_runs(),
        runs + 1,
        "the second text ran the shared plan"
    );
}
