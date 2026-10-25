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

//! A statement's per-row programs, with its parameters and uncorrelated
//! blocks bound to the values one execution reads, are op for op the
//! programs of the statement with those values written as literals. A
//! parameter or run slot left to load its value per row, which costs what
//! loading the constant does, counts as that constant.

#![cfg(feature = "test-failpoints")]

use stoolap::core::Value;
use stoolap::test_failpoints::programs_ran;
use stoolap::Database;

/// A program the VM entered at least this often ran per row or per group
const PER_ROW: usize = 3;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, s TEXT)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, y TEXT)",
        "INSERT INTO t SELECT value, value % 8, value % 5, 'v' || CAST(value AS TEXT) \
         FROM generate_series(1, 40)",
        "INSERT INTO u VALUES (1, 6, 'v1*')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The per-row programs one execution of `sql` with `params` ran
fn row_programs(db: &Database, sql: &str, params: &[Value]) -> Result<Vec<String>, String> {
    let stmt = db.prepare(sql).map_err(|e| e.to_string())?;
    let run = |params: Vec<Value>| -> Result<(), String> {
        if sql.starts_with("SELECT") {
            for row in stmt.query(params).map_err(|e| e.to_string())? {
                row.map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        stmt.execute(params).map(|_| ()).map_err(|e| e.to_string())
    };
    // A first execution, with other values, compiles and caches what the
    // plan keeps: what it bound must not reach the next
    let other = params
        .iter()
        .map(|value| match value {
            Value::Integer(n) => Value::Integer(n + 100),
            Value::Text(s) => Value::text(format!("{s}z")),
            value => value.clone(),
        })
        .collect();
    run(other)?;
    let mut failed = None;
    let ran = programs_ran(|| failed = run(params.to_vec()).err());
    match failed {
        Some(e) => Err(e),
        None => {
            let mut listings: Vec<String> = Vec::new();
            for (listing, times) in ran {
                let listing = as_constants(listing, params);
                if times >= PER_ROW && !listings.contains(&listing) {
                    listings.push(listing);
                }
            }
            Ok(listings)
        }
    }
}

/// `listing` with each parameter load read as the load of its value, and a
/// NULL constant as a NULL literal loads
fn as_constants(mut listing: String, params: &[Value]) -> String {
    for (slot, value) in params.iter().enumerate() {
        listing = listing.replace(
            &format!("LoadParam({slot})\n"),
            &format!("LoadConst({value:?})\n"),
        );
    }
    listing.replace("LoadConst(Null(Null))", "LoadNull(Null)")
}

/// Whether `ran` are `literal` op for op, a run slot's load, which costs
/// what loading its value does, counting as the constant's
fn same_programs(ran: &[String], literal: &[String]) -> bool {
    let load = |op: &str| op.starts_with("LoadConst(") || op.starts_with("LoadNull(");
    ran.len() == literal.len()
        && ran.iter().zip(literal).all(|(ran, literal)| {
            ran.lines().count() == literal.lines().count()
                && ran.lines().zip(literal.lines()).all(|(r, l)| {
                    let (r, l) = (&r[6..], &l[6..]);
                    r == l
                        || (r.starts_with("LoadParam(") && (load(l) || l.starts_with("LoadParam(")))
                })
        })
}

#[test]
fn a_bound_statement_runs_its_literal_twins_programs() {
    let db = open("prepared_literal_twin");
    let int = Value::Integer;
    let text = |s: &str| Value::text(s);
    // (statement, its values, the statement with the values as literals)
    let shapes: Vec<(&str, Vec<Value>, &str)> = vec![
        (
            "SELECT COUNT(*) FROM t WHERE (a = $1) IS TRUE",
            vec![int(6)],
            "SELECT COUNT(*) FROM t WHERE (a = 6) IS TRUE",
        ),
        (
            "SELECT id FROM t WHERE (a = $1) IS TRUE",
            vec![int(6)],
            "SELECT id FROM t WHERE (a = 6) IS TRUE",
        ),
        (
            "SELECT id FROM t WHERE s GLOB $1",
            vec![text("v1*")],
            "SELECT id FROM t WHERE s GLOB 'v1*'",
        ),
        (
            "SELECT id FROM t WHERE s LIKE $1",
            vec![text("v1%")],
            "SELECT id FROM t WHERE s LIKE 'v1%'",
        ),
        (
            "SELECT id FROM t WHERE s ILIKE $1",
            vec![text("V1%")],
            "SELECT id FROM t WHERE s ILIKE 'V1%'",
        ),
        (
            "SELECT id FROM t WHERE s LIKE $1 ESCAPE '!'",
            vec![text("v1!%")],
            "SELECT id FROM t WHERE s LIKE 'v1!%' ESCAPE '!'",
        ),
        (
            "SELECT id FROM t WHERE a + $1 = 8",
            vec![int(2)],
            "SELECT id FROM t WHERE a + 2 = 8",
        ),
        (
            "SELECT id, a = $1 + 2 FROM t",
            vec![int(4)],
            "SELECT id, a = 4 + 2 FROM t",
        ),
        (
            "SELECT id, s LIKE $1 || '%' FROM t",
            vec![text("v1")],
            "SELECT id, s LIKE 'v1' || '%' FROM t",
        ),
        (
            "SELECT id, a IN ($1, 3, $2) FROM t",
            vec![int(6), int(1)],
            "SELECT id, a IN (6, 3, 1) FROM t",
        ),
        (
            "SELECT id, COALESCE($1, a) + b FROM t",
            vec![Value::null_unknown()],
            "SELECT id, COALESCE(NULL, a) + b FROM t",
        ),
        (
            "SELECT id, a NOT IN ($1, NULL) FROM t",
            vec![int(6)],
            "SELECT id, a NOT IN (6, NULL) FROM t",
        ),
        (
            "SELECT id, s LIKE $1 ESCAPE $2 FROM t",
            vec![text("v1!%"), text("!")],
            "SELECT id, s LIKE 'v1!%' ESCAPE '!' FROM t",
        ),
        (
            "SELECT id, a < ABS($1 - 9) FROM t",
            vec![int(4)],
            "SELECT id, a < ABS(4 - 9) FROM t",
        ),
        (
            "SELECT id FROM t WHERE s GLOB UPPER($1) OR a = CAST($2 AS INTEGER) * 2",
            vec![text("v1*"), text("3")],
            "SELECT id FROM t WHERE s GLOB UPPER('v1*') OR a = CAST('3' AS INTEGER) * 2",
        ),
        (
            "SELECT id, s || ($1 || $2) FROM t",
            vec![text("x"), text("y")],
            "SELECT id, s || ('x' || 'y') FROM t",
        ),
        (
            "SELECT id FROM t WHERE a = $1 / 0 OR b = 1",
            vec![int(4)],
            "SELECT id FROM t WHERE a = 4 / 0 OR b = 1",
        ),
        (
            "SELECT id FROM t WHERE a < $1 + RANDOM()",
            vec![int(4)],
            "SELECT id FROM t WHERE a < 4 + RANDOM()",
        ),
        (
            "SELECT id FROM t WHERE a + 1 = $1",
            vec![text("7")],
            "SELECT id FROM t WHERE a + 1 = '7'",
        ),
        (
            "SELECT id, s || $1 FROM t",
            vec![text("x")],
            "SELECT id, s || 'x' FROM t",
        ),
        (
            "SELECT id, CASE WHEN a > $1 THEN b ELSE 0 END FROM t",
            vec![int(3)],
            "SELECT id, CASE WHEN a > 3 THEN b ELSE 0 END FROM t",
        ),
        (
            "SELECT a, COUNT(*) FROM t WHERE b + $1 > 2 GROUP BY a",
            vec![int(1)],
            "SELECT a, COUNT(*) FROM t WHERE b + 1 > 2 GROUP BY a",
        ),
        (
            "SELECT a, SUM(b) FROM t GROUP BY a HAVING SUM(b) + $1 > 4",
            vec![int(1)],
            "SELECT a, SUM(b) FROM t GROUP BY a HAVING SUM(b) + 1 > 4",
        ),
        (
            "SELECT id FROM t ORDER BY a + $1, id",
            vec![int(1)],
            "SELECT id FROM t ORDER BY a + 1, id",
        ),
        (
            "SELECT t.id FROM t JOIN u ON t.a = u.x AND t.b + $1 > 2",
            vec![int(1)],
            "SELECT t.id FROM t JOIN u ON t.a = u.x AND t.b + 1 > 2",
        ),
        (
            "SELECT t.id, t.b + $1 FROM t JOIN u ON t.b <= u.x ORDER BY t.id",
            vec![int(1)],
            "SELECT t.id, t.b + 1 FROM t JOIN u ON t.b <= u.x ORDER BY t.id",
        ),
        (
            "SELECT t.id FROM t JOIN u ON t.b <= u.x ORDER BY t.b * $1, t.id",
            vec![int(2)],
            "SELECT t.id FROM t JOIN u ON t.b <= u.x ORDER BY t.b * 2, t.id",
        ),
        (
            "SELECT SUM(a * $1) FROM t",
            vec![int(2)],
            "SELECT SUM(a * 2) FROM t",
        ),
        (
            "SELECT a + $1, COUNT(*) FROM t GROUP BY a + $1",
            vec![int(1)],
            "SELECT a + 1, COUNT(*) FROM t GROUP BY a + 1",
        ),
        (
            "SELECT id, SUM(b) OVER (ORDER BY a + $1, id) FROM t",
            vec![int(1)],
            "SELECT id, SUM(b) OVER (ORDER BY a + 1, id) FROM t",
        ),
        (
            "SELECT x FROM (SELECT a + $1 AS x FROM t) d",
            vec![int(1)],
            "SELECT x FROM (SELECT a + 1 AS x FROM t) d",
        ),
        (
            "WITH c AS (SELECT a + $1 AS x FROM t) SELECT x FROM c",
            vec![int(1)],
            "WITH c AS (SELECT a + 1 AS x FROM t) SELECT x FROM c",
        ),
        (
            "UPDATE t SET b = b WHERE a IN (SELECT x FROM u) OR a < 9 RETURNING b + $1",
            vec![int(1)],
            "UPDATE t SET b = b WHERE a IN (SELECT x FROM u) OR a < 9 RETURNING b + 1",
        ),
        (
            "SELECT t.id, t.b + $1 FROM t JOIN u ON t.b <= u.x \
             WHERE t.a IN (SELECT x FROM u) OR t.a < 9",
            vec![int(1)],
            "SELECT t.id, t.b + 1 FROM t JOIN u ON t.b <= u.x \
             WHERE t.a IN (SELECT x FROM u) OR t.a < 9",
        ),
        (
            "UPDATE t SET b = b WHERE a + 1 > 0 RETURNING b + $1",
            vec![int(1)],
            "UPDATE t SET b = b WHERE a + 1 > 0 RETURNING b + 1",
        ),
        (
            "UPDATE t SET b = b WHERE a + $1 = 8",
            vec![int(2)],
            "UPDATE t SET b = b WHERE a + 2 = 8",
        ),
        (
            "DELETE FROM t WHERE a + $1 = 100",
            vec![int(2)],
            "DELETE FROM t WHERE a + 2 = 100",
        ),
        (
            "SELECT COUNT(*) FROM t WHERE (a = (SELECT MAX(x) FROM u)) IS TRUE",
            vec![],
            "SELECT COUNT(*) FROM t WHERE (a = 6) IS TRUE",
        ),
        (
            "SELECT id FROM t WHERE s GLOB (SELECT y FROM u WHERE id = 1)",
            vec![],
            "SELECT id FROM t WHERE s GLOB 'v1*'",
        ),
        (
            "SELECT id FROM t WHERE b & (SELECT MAX(x) FROM u) = 2",
            vec![],
            "SELECT id FROM t WHERE b & 6 = 2",
        ),
        (
            "SELECT id FROM t WHERE a IN (SELECT x FROM u) AND b + $1 > 2",
            vec![int(1)],
            "SELECT id FROM t WHERE a IN (SELECT x FROM u) AND b + 1 > 2",
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = t.a) AND s GLOB $1",
            vec![text("v1*")],
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = t.a) AND s GLOB 'v1*'",
        ),
        (
            "SELECT id, (SELECT MAX(x) FROM u) + a * $1 FROM t",
            vec![int(2)],
            "SELECT id, (SELECT MAX(x) FROM u) + a * 2 FROM t",
        ),
        (
            "SELECT id FROM t WHERE a = (SELECT MAX(x) FROM u) OR b + $1 = 4",
            vec![int(1)],
            "SELECT id FROM t WHERE a = (SELECT MAX(x) FROM u) OR b + 1 = 4",
        ),
        (
            "UPDATE t SET b = b WHERE a IN (SELECT x FROM u) AND b + $1 > 2",
            vec![int(1)],
            "UPDATE t SET b = b WHERE a IN (SELECT x FROM u) AND b + 1 > 2",
        ),
        (
            "DELETE FROM t WHERE a IN (SELECT x FROM u) AND b + $1 > 100",
            vec![int(1)],
            "DELETE FROM t WHERE a IN (SELECT x FROM u) AND b + 1 > 100",
        ),
        (
            "SELECT id FROM t WHERE a = (SELECT MAX(x) FROM u) OR b + $1 = 4",
            vec![int(1)],
            "SELECT id FROM t WHERE a = 6 OR b + 1 = 4",
        ),
        (
            "SELECT (SELECT MAX(x) FROM u) + a * $1 FROM t",
            vec![int(2)],
            "SELECT 6 + a * 2 FROM t",
        ),
        (
            "SELECT id FROM t WHERE s GLOB (SELECT y FROM u WHERE id = 1) OR a = $1",
            vec![int(3)],
            "SELECT id FROM t WHERE s GLOB 'v1*' OR a = 3",
        ),
        (
            "UPDATE t SET b = b WHERE a = (SELECT MAX(x) FROM u) OR b + $1 = 4",
            vec![int(1)],
            "UPDATE t SET b = b WHERE a = 6 OR b + 1 = 4",
        ),
    ];
    let mut wrong = Vec::new();
    for (sql, params, twin) in &shapes {
        let bound = row_programs(&db, sql, params);
        let literal = row_programs(&db, twin, &[]);
        let same = match (&bound, &literal) {
            (Ok(bound), Ok(literal)) => same_programs(bound, literal),
            (bound, literal) => bound == literal,
        };
        if !same {
            wrong.push(format!(
                "{sql} {params:?}\n  ran     {bound:?}\n  literal {literal:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
