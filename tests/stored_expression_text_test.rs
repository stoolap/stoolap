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

//! An expression stored as text, a DEFAULT, a CHECK or a view, reads back as
//! the expression it was: a string literal keeps its value whatever quote or
//! backslash it holds.

use stoolap::{Database, Value};

const VALUES: [&str; 8] = [
    "it's",
    "''",
    "a\\",
    "a\\b",
    "x\\n",
    "tab\there",
    "{\"k\":\"v \\\"q\\\"\"}",
    "plain",
];

/// `value` as a string literal: a quote and a backslash doubled
fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "''"))
}

/// The first column of the first row of `sql`
fn one(db: &Database, sql: &str) -> Result<Value, String> {
    db.query(sql, ())
        .map_err(|e| e.to_string())?
        .next()
        .ok_or_else(|| "no row".to_string())?
        .map(|row| row.get_value(0).cloned().unwrap())
        .map_err(|e| e.to_string())
}

/// `write` run, then `read` answered, as one result
fn after(db: &Database, write: &str, read: &str) -> Result<Value, String> {
    db.execute(write, ()).map_err(|e| e.to_string())?;
    one(db, read)
}

fn wrong_answers(n: usize, value: &str, wrong: &mut Vec<String>) {
    let lit = literal(value);
    let want = Value::text(value);
    let db = Database::open(&format!("memory://stored_text_{n}")).unwrap();
    db.execute(
        &format!(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT DEFAULT {lit}, c TEXT CHECK (c <> {lit}))"
        ),
        (),
    )
    .unwrap();
    let mut answers = vec![(
        "DEFAULT",
        after(
            &db,
            "INSERT INTO t (id) VALUES (1)",
            "SELECT s FROM t WHERE id = 1",
        ),
    )];
    if db
        .execute(&format!("INSERT INTO t (id, c) VALUES (2, {lit})"), ())
        .is_ok()
    {
        wrong.push(format!("{value:?} CHECK let its own value in"));
    }
    if let Err(e) = db.execute("INSERT INTO t (id, c) VALUES (3, 'other')", ()) {
        wrong.push(format!("{value:?} CHECK refused another value: {e}"));
    }
    answers.push((
        "view",
        after(
            &db,
            &format!("CREATE VIEW v AS SELECT s FROM t WHERE s = {lit} AND id = 1"),
            "SELECT s FROM v",
        ),
    ));
    answers.push((
        "ALTER ADD COLUMN, an existing row",
        after(
            &db,
            &format!("ALTER TABLE t ADD COLUMN d TEXT DEFAULT {lit}"),
            "SELECT d FROM t WHERE id = 1",
        ),
    ));
    answers.push((
        "ALTER ADD COLUMN, a new row",
        after(
            &db,
            "INSERT INTO t (id) VALUES (4)",
            "SELECT d FROM t WHERE id = 4",
        ),
    ));
    let shown = db
        .query("SHOW CREATE TABLE t", ())
        .map_err(|e| e.to_string())
        .and_then(|mut rows| {
            let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
            row.get::<String>(1).map_err(|e| e.to_string())
        });
    let copy = Database::open(&format!("memory://stored_text_copy_{n}")).unwrap();
    answers.push((
        "SHOW CREATE TABLE run again",
        shown.and_then(|ddl| {
            copy.execute(&ddl, ()).map_err(|e| format!("{ddl}: {e}"))?;
            after(
                &copy,
                "INSERT INTO t (id) VALUES (1)",
                "SELECT s FROM t WHERE id = 1",
            )
        }),
    ));
    for (context, got) in answers {
        if got.as_ref() != Ok(&want) {
            wrong.push(format!("{value:?} {context}: gave {got:?}"));
        }
    }
}

#[test]
fn test_a_stored_expression_reads_back_as_written() {
    let mut wrong = Vec::new();
    for (n, value) in VALUES.iter().enumerate() {
        wrong_answers(n, value, &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
