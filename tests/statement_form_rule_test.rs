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

//! A window function stands only in a SELECT list, DISTINCT ON or ORDER BY and
//! outside any other call's input, and ANY or ALL take only a comparison
//! operator: anything else is the statement's error before it runs, in every
//! clause and statement, hot and sealed, prepared or not

use stoolap::{Database, Error, Value};

const OUTSIDE: &str = "is allowed only in a SELECT list, DISTINCT ON or ORDER BY";
const NESTED: &str = "cannot be read inside a call of";
const QUANTIFIED: &str = "takes a comparison operator (=, <>, <, <=, >, >=), not";

/// Each statement the rule refuses, the error it raises, and its parameters
const REFUSED: &[(&str, &str, &[i64])] = &[
    ("SELECT a FROM t WHERE ROW_NUMBER() OVER () > 1", OUTSIDE, &[]),
    ("SELECT a FROM t WHERE ROW_NUMBER() OVER () > $1", OUTSIDE, &[1]),
    ("SELECT g FROM t GROUP BY g HAVING SUM(a) > ROW_NUMBER() OVER ()", OUTSIDE, &[]),
    ("SELECT g FROM t GROUP BY ROW_NUMBER() OVER ()", OUTSIDE, &[]),
    ("SELECT a FROM t LIMIT ROW_NUMBER() OVER ()", OUTSIDE, &[]),
    ("SELECT a FROM t OFFSET ROW_NUMBER() OVER ()", OUTSIDE, &[]),
    ("SELECT * FROM t JOIN u ON ROW_NUMBER() OVER () = u.id", OUTSIDE, &[]),
    ("SELECT * FROM generate_series(1, ROW_NUMBER() OVER ())", OUTSIDE, &[]),
    ("SELECT CASE WHEN a > 1 THEN 1 END FROM t WHERE CASE WHEN ROW_NUMBER() OVER () > 1 THEN TRUE END", OUTSIDE, &[]),
    ("SELECT a FROM t WHERE a IN (SELECT tid FROM u WHERE ROW_NUMBER() OVER () > 1)", OUTSIDE, &[]),
    ("SELECT a FROM t WHERE EXISTS (SELECT 1 FROM u WHERE ROW_NUMBER() OVER () > 1)", OUTSIDE, &[]),
    ("SELECT a, ROW_NUMBER() OVER w FROM t WINDOW w AS (PARTITION BY ROW_NUMBER() OVER ())", OUTSIDE, &[]),
    ("SELECT SUM(ROW_NUMBER() OVER (ORDER BY a)) FROM t", NESTED, &[]),
    ("SELECT COUNT(*) FILTER (WHERE ROW_NUMBER() OVER () > 1) FROM t", NESTED, &[]),
    ("SELECT SUM(ROW_NUMBER() OVER (ORDER BY a)) OVER () FROM t", NESTED, &[]),
    ("SELECT ROW_NUMBER() OVER (PARTITION BY ROW_NUMBER() OVER ()) FROM t", NESTED, &[]),
    ("SELECT ROW_NUMBER() OVER (ORDER BY ROW_NUMBER() OVER ()) FROM t", NESTED, &[]),
    ("SELECT LAG(a, ROW_NUMBER() OVER ()) OVER (ORDER BY a) FROM t", NESTED, &[]),
    ("UPDATE t SET b = ROW_NUMBER() OVER () WHERE id = 1", OUTSIDE, &[]),
    ("UPDATE t SET b = 1 WHERE ROW_NUMBER() OVER () = $1", OUTSIDE, &[1]),
    ("DELETE FROM t WHERE ROW_NUMBER() OVER () = 99", OUTSIDE, &[]),
    ("INSERT INTO u VALUES (100, ROW_NUMBER() OVER ())", OUTSIDE, &[]),
    ("UPDATE t SET b = b WHERE id = 1 RETURNING ROW_NUMBER() OVER ()", OUTSIDE, &[]),
    ("INSERT INTO u VALUES (101, 1) RETURNING ROW_NUMBER() OVER ()", OUTSIDE, &[]),
    ("INSERT INTO u VALUES (1, 1) ON CONFLICT (id) DO UPDATE SET tid = ROW_NUMBER() OVER ()", OUTSIDE, &[]),
    ("SELECT a + ANY (SELECT 1) FROM t", QUANTIFIED, &[]),
    ("SELECT a FROM t WHERE a + ANY (SELECT tid FROM u) > $1", QUANTIFIED, &[0]),
    ("SELECT a - ALL (SELECT 1) FROM t", QUANTIFIED, &[]),
    ("SELECT CASE WHEN a * ANY (SELECT 2) > 2 THEN 1 END FROM t", QUANTIFIED, &[]),
    ("SELECT g FROM t GROUP BY g HAVING SUM(a) / ALL (SELECT 2) > 1", QUANTIFIED, &[]),
    ("SELECT a FROM t WHERE (a || ANY (SELECT 'x')) IS NULL", QUANTIFIED, &[]),
    ("SELECT a FROM t WHERE a % SOME (SELECT 2) = 0", QUANTIFIED, &[]),
    ("UPDATE t SET b = b WHERE a + ANY (SELECT 1) > 0", QUANTIFIED, &[]),
    ("DELETE FROM t WHERE a + ANY (SELECT 1) > 0", QUANTIFIED, &[]),
];

/// Each statement the rule lets through, with its parameters and its answer
const ANSWERED: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT a, ROW_NUMBER() OVER (ORDER BY a) FROM t WHERE a > $1 ORDER BY a",
        &[3],
        &["Integer(4)|Integer(1)", "Integer(5)|Integer(2)"],
    ),
    (
        "SELECT a FROM t ORDER BY ROW_NUMBER() OVER (ORDER BY a DESC) LIMIT 2",
        &[],
        &["Integer(5)", "Integer(4)"],
    ),
    (
        "SELECT CASE WHEN ROW_NUMBER() OVER (ORDER BY a) > 4 THEN a ELSE 0 END FROM t ORDER BY a",
        &[],
        &["Integer(0)", "Integer(0)", "Integer(0)", "Integer(0)", "Integer(5)"],
    ),
    (
        "SELECT g, SUM(SUM(a)) OVER () FROM t GROUP BY g ORDER BY g",
        &[],
        &["Integer(0)|Integer(15)", "Integer(1)|Integer(15)"],
    ),
    (
        "SELECT DISTINCT ON (ROW_NUMBER() OVER (ORDER BY id) % 2) id, ROW_NUMBER() OVER (ORDER BY id) % 2 AS k FROM t ORDER BY id",
        &[],
        &["Integer(1)|Integer(1)", "Integer(2)|Integer(0)"],
    ),
    (
        "SELECT (SELECT tid FROM u ORDER BY ROW_NUMBER() OVER (ORDER BY tid DESC) LIMIT 1) FROM t WHERE id = 1",
        &[],
        &["Integer(3)"],
    ),
    (
        "SELECT id FROM t WHERE id < 3 WINDOW w AS (ORDER BY ROW_NUMBER() OVER ()) ORDER BY id",
        &[],
        &["Integer(1)", "Integer(2)"],
    ),
    (
        "SELECT id, ROW_NUMBER() OVER (w ORDER BY id DESC) FROM t WHERE id < 3 \
         WINDOW w AS (ORDER BY ROW_NUMBER() OVER ()) ORDER BY id",
        &[],
        &["Integer(1)|Integer(2)", "Integer(2)|Integer(1)"],
    ),
    (
        "SELECT a FROM t WHERE a != ALL (SELECT tid FROM u) ORDER BY a",
        &[],
        &["Integer(4)", "Integer(5)"],
    ),
    (
        "SELECT a FROM t WHERE a <= SOME (SELECT tid FROM u WHERE tid < $1) ORDER BY a",
        &[3],
        &["Integer(1)", "Integer(2)"],
    ),
];

/// t and u; sealed into volumes when `cold`
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
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, g INTEGER)",
        "INSERT INTO t SELECT value, value, value * 2, value % 2 FROM generate_series(1, 5)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER)",
        "INSERT INTO u SELECT value, value FROM generate_series(1, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if cold {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn values(params: &[i64]) -> Vec<Value> {
    params.iter().map(|p| Value::Integer(*p)).collect()
}

/// Every row of `sql`, run as given or prepared first
fn rows(db: &Database, sql: &str, params: &[i64], prepared: bool) -> stoolap::Result<Vec<String>> {
    let rows = match prepared {
        true => db.prepare(sql)?.query(values(params))?,
        false => db.query(sql, values(params))?,
    };
    rows.map(|row| {
        let row = row?;
        Ok((0..row.len())
            .map(|i| format!("{:?}", row.get_value(i).unwrap()))
            .collect::<Vec<_>>()
            .join("|"))
    })
    .collect()
}

#[test]
fn a_form_no_path_answers_is_the_statements_error_everywhere() {
    let mut wrong = Vec::new();
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = tables(&format!("statement_form_rule_{cold}"), cold, &dir);
        for prepared in [false, true] {
            for &(sql, error, params) in REFUSED {
                match rows(&db, sql, params, prepared) {
                    Err(Error::InvalidArgument(message)) if message.contains(error) => {}
                    got => wrong.push(format!(
                        "{sql} (sealed {cold}, prepared {prepared}): {got:?}, expected {error}"
                    )),
                }
            }
            for &(sql, params, answer) in ANSWERED {
                match rows(&db, sql, params, prepared) {
                    Ok(got) if got == answer => {}
                    got => wrong.push(format!(
                        "{sql} (sealed {cold}, prepared {prepared}): {got:?}, expected {answer:?}"
                    )),
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
