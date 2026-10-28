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

//! A join with a subquery run inside a statement that holds a plan keeps
//! its own plan there: it runs no general path mechanism, run again it
//! compiles nothing, and a table made again with its columns moved
//! prepares it again

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{bridge_compiles, general_path_hits, program_compiles, GeneralPath};
use stoolap::Database;

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const ALL: [GeneralPath; 15] = [
    GeneralPath::CorrelatedScan,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::PushdownConstant,
    GeneralPath::ProjectionEvaluator,
    GeneralPath::OrderByEvaluator,
    GeneralPath::AntiJoin,
    GeneralPath::ExistsSemiJoin,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
    GeneralPath::ScalarSubquery,
    GeneralPath::SelectFold,
    GeneralPath::CorrelatedExpression,
    GeneralPath::LiftedSubqueries,
];

fn hits() -> usize {
    ALL.iter().map(|path| general_path_hits(*path)).sum()
}

const JOIN: &str =
    "SELECT t.id, u.x FROM t JOIN u ON u.id = t.a WHERE t.b > (SELECT MAX(x) FROM u)";

/// Each shape, `{j}` standing for the join
const SHAPES: &[(&str, &str)] = &[
    (
        "derived table of a prepared statement",
        "SELECT d.id FROM ({j}) AS d WHERE d.id > (SELECT MIN(x) FROM u)",
    ),
    ("view", "SELECT * FROM vj"),
    (
        "view of a prepared statement",
        "SELECT vj.id FROM vj WHERE vj.id > (SELECT MIN(x) FROM u)",
    ),
    (
        "view as a block source",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM vj WHERE vj.id = t.id)",
    ),
    (
        "CTE of a prepared statement",
        "WITH c AS ({j}) SELECT c.id FROM c WHERE c.id > (SELECT MIN(x) FROM u)",
    ),
    (
        "CTE read twice",
        "WITH c AS ({j}) SELECT a.id FROM c AS a JOIN c AS b ON a.id = b.id \
         WHERE a.id > (SELECT MIN(x) FROM u)",
    ),
    (
        "derived join side",
        "SELECT t.id FROM t JOIN ({j}) AS d ON d.id = t.id WHERE t.b > (SELECT MIN(x) FROM u)",
    ),
    (
        "set operation branch",
        "SELECT id, x FROM u WHERE id > (SELECT 0) UNION {j}",
    ),
    (
        "recursive CTE's first member",
        "WITH RECURSIVE r(id, x) AS ({j} UNION ALL \
         SELECT r.id + 10, u.x FROM r JOIN u ON u.id = r.x WHERE r.id < (SELECT 10)) \
         SELECT r.id FROM r WHERE r.id > (SELECT MIN(x) FROM u) - 1",
    ),
    (
        "recursive CTE's recursive member",
        "WITH RECURSIVE r(id, x) AS (SELECT id, x FROM u WHERE id = (SELECT 1) UNION ALL \
         SELECT u.id, u.x FROM r JOIN u ON u.id = r.id + 1 WHERE u.x > (SELECT MIN(x) FROM u) - 1) \
         SELECT r.id, r.x FROM r WHERE r.id > (SELECT MIN(x) FROM u) - 1",
    ),
    (
        "CTE its LIMIT cuts, kept by a LEFT JOIN",
        "WITH c (a, n) AS (SELECT t.a, COUNT(*) FROM t JOIN u ON u.id = t.id \
         WHERE t.b > (SELECT MIN(x) FROM u) GROUP BY t.a) \
         SELECT c.a, u.x FROM c LEFT JOIN u ON u.id = c.a AND u.x < (SELECT MAX(x) FROM u) \
         ORDER BY c.a LIMIT 2",
    ),
    (
        "CTE its LIMIT cuts and grows under an INNER JOIN",
        "WITH c (a, n) AS (SELECT t.a, COUNT(*) FROM t JOIN u ON u.id = t.id \
         WHERE t.b > (SELECT MIN(x) FROM u) GROUP BY t.a) \
         SELECT c.a, u.x FROM c JOIN u ON u.id = c.a AND u.x >= (SELECT MIN(x) FROM u) \
         ORDER BY c.a LIMIT 1",
    ),
];

/// Each statement that runs `{j}` as a part, the statement after it
/// reading what it left
const STATEMENTS: &[(&str, &str, &str)] = &[
    ("INSERT ... SELECT", "INSERT INTO w {j}", "SELECT * FROM w"),
    (
        "UPDATE's CTE",
        "WITH c AS ({j}) UPDATE t SET b = b + 1 WHERE id IN (SELECT id FROM c)",
        "SELECT * FROM t",
    ),
    (
        "DELETE's CTE",
        "WITH c AS ({j}) DELETE FROM w WHERE id IN (SELECT id FROM c) AND x > 2",
        "SELECT * FROM w",
    ),
    ("EXPLAIN ANALYZE", "EXPLAIN ANALYZE {j}", "SELECT * FROM t"),
];

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://nested_plan_slots_{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 0, 30), (4, 1, 40), (5, 2, 50), (6, 0, 60)",
        "INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)",
        "CREATE TABLE w (id INTEGER, x INTEGER)",
        &format!("CREATE VIEW vj AS {JOIN}"),
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get::<String>(i).unwrap())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    out.sort();
    out
}

fn compiles() -> (usize, usize) {
    (bridge_compiles(), program_compiles())
}

#[test]
fn a_nested_join_runs_as_blocks_and_run_again_compiles_nothing() {
    let _guard = serial();
    let db = setup("again");
    let mut wrong = Vec::new();
    for (name, shape) in SHAPES {
        let sql = shape.replace("{j}", JOIN);
        let hit = hits();
        let first = rows(&db, &sql);
        let before = compiles();
        let again = rows(&db, &sql);
        if hits() != hit {
            wrong.push(format!("{name}: ran a general path mechanism"));
        }
        if compiles() != before || again != first {
            wrong.push(format!(
                "{name}: run again compiled {:?} -> {:?}, gave {first:?} then {again:?}",
                before,
                compiles()
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_join_a_statement_runs_as_a_part_runs_as_blocks_and_run_again_compiles_nothing() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for (index, (name, statement, read)) in STATEMENTS.iter().enumerate() {
        let db = setup(&format!("statement_{index}"));
        let sql = statement.replace("{j}", JOIN);
        let hit = hits();
        db.execute(&sql, ())
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let first = rows(&db, read);
        let before = compiles();
        db.execute(&sql, ())
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        if hits() != hit {
            wrong.push(format!("{name}: ran a general path mechanism"));
        }
        if compiles() != before {
            wrong.push(format!(
                "{name}: run again compiled {:?} -> {:?}, left {first:?}",
                before,
                compiles()
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn create_table_as_a_join_runs_it_as_blocks_and_made_again_compiles_nothing() {
    let _guard = serial();
    let db = setup("create_table_as");
    let create = format!("CREATE TABLE c AS {JOIN}");
    let hit = hits();
    db.execute(&create, ()).unwrap();
    assert_eq!(hits(), hit, "ran a general path mechanism");
    assert_eq!(rows(&db, "SELECT * FROM c"), rows(&db, JOIN));
    db.execute("DROP TABLE c", ()).unwrap();
    let before = compiles();
    db.execute(&create, ()).unwrap();
    assert_eq!(compiles(), before, "made again, it compiles nothing");
}

#[test]
fn a_nested_join_over_a_table_made_again_reads_its_columns_as_they_now_are() {
    let _guard = serial();
    let mut wrong = Vec::new();
    for (index, (name, shape)) in SHAPES.iter().enumerate() {
        let db = setup(&format!("remade_{index}"));
        let sql = shape.replace("{j}", JOIN);
        let first = rows(&db, &sql);
        for remake in [
            "DROP TABLE u",
            "CREATE TABLE u (x INTEGER, id INTEGER PRIMARY KEY)",
            "INSERT INTO u (id, x) VALUES (1, 1), (2, 2), (3, 3)",
        ] {
            db.execute(remake, ()).unwrap();
        }
        let after = rows(&db, &sql);
        if after != first {
            wrong.push(format!("{name}: gave {first:?}, then {after:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
