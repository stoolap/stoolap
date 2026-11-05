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

//! An expression that fails to compile is the statement's error: it is
//! never read as NULL or zero and never dropped from a SET or a join.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "CREATE INDEX ux ON u(x)",
        "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20)",
        "INSERT INTO u VALUES (1, 1), (2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<Vec<String>>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect(),
        );
    }
    Ok(out)
}

#[test]
fn test_update_set_that_fails_to_compile() {
    let db = setup("compile_error_set");
    let outcome = db.execute("UPDATE t SET b = NO_SUCH_FN(a) WHERE id = 1", ());
    assert!(
        outcome.is_err(),
        "UPDATE with an unknown function: {outcome:?}"
    );
    assert_eq!(
        rows(&db, "SELECT b FROM t WHERE id = 1"),
        Ok(vec![vec!["10".to_string()]])
    );
}

#[test]
fn test_join_residual_that_fails_to_compile() {
    let db = setup("compile_error_join_on");
    let got = rows(
        &db,
        "SELECT t.id, u.id FROM t JOIN u ON u.x = t.a AND NO_SUCH_FN(u.id) = 1 ORDER BY t.id",
    );
    assert!(got.is_err(), "ON with an unknown function: {got:?}");
}

#[test]
fn test_offset_that_fails() {
    let db = setup("compile_error_offset");
    for sql in [
        "SELECT id FROM t ORDER BY id OFFSET UPPER('a', 'b')",
        "SELECT id FROM t ORDER BY id OFFSET (SELECT 1)",
        "SELECT id FROM t ORDER BY id OFFSET 9223372036854775807 + 1",
    ] {
        let got = rows(&db, sql);
        assert!(
            got.is_err() || got == Ok(vec![vec!["2".to_string()]]),
            "{sql}: {got:?}"
        );
    }
}

#[test]
fn test_default_that_fails_to_compile() {
    let db = setup("compile_error_default");
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, v INTEGER DEFAULT (nope + 5))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO d VALUES (1, 7)", ())
        .expect("a row that names the column takes no default");
    let _ = db.execute("INSERT INTO d (id) VALUES (2)", ());
    assert_ne!(
        rows(&db, "SELECT v FROM d WHERE id = 2"),
        Ok(vec![vec!["NULL".to_string()]]),
        "a default read as NULL"
    );
}
