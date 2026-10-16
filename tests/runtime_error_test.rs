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

//! A runtime error in an expression over rows fails the statement wherever
//! the expression sits, and changes no row; a NULL input stays a NULL

use stoolap::Database;

const BIG: &str = "9223372036854775807";

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://runtime_error_{name}")).unwrap();
    for sql in [
        "CREATE TABLE v (id INTEGER PRIMARY KEY, n INTEGER, g INTEGER, z INTEGER)",
        "INSERT INTO v VALUES (1, 3, 1, NULL), (2, 5, 1, NULL)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, n INTEGER)",
        "INSERT INTO w VALUES (1, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// Every row `sql` returns, or the first error
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for row in db.query(sql, ()).map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get_value(i).unwrap() {
                    v if v.is_null() => "NULL".to_string(),
                    v => format!("{v:?}"),
                })
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok(out)
}

#[test]
fn a_runtime_error_in_a_select_fails_it_wherever_it_sits() {
    let db = open("select");
    let mut wrong = Vec::new();
    for sql in [
        format!("SELECT {BIG} + n FROM v"),
        format!("SELECT id, {BIG} + n FROM v WHERE id = 1"),
        format!("SELECT CASE WHEN n > 0 THEN {BIG} + n END FROM v"),
        format!("SELECT x FROM (SELECT {BIG} + n AS x FROM v) s"),
        format!("SELECT {BIG} + n FROM v UNION ALL SELECT 1"),
        format!("SELECT id FROM v ORDER BY {BIG} + n"),
        format!("SELECT id FROM v ORDER BY {BIG} + n LIMIT 1"),
        format!("SELECT id FROM v ORDER BY {BIG} + n DESC, id"),
        format!("SELECT DISTINCT ON ({BIG} + n) id FROM v"),
        format!(
            "SELECT id FROM v WHERE n > (SELECT n FROM w WHERE w.id = v.id) - 100 ORDER BY {BIG} + n"
        ),
        format!(
            "SELECT id, n AS m FROM v WHERE n > (SELECT n FROM w WHERE w.id = v.id) - 100 ORDER BY m + {BIG}"
        ),
        format!("SELECT COUNT(*) FROM v GROUP BY {BIG} + n"),
        format!("SELECT g, COUNT(*) FROM v GROUP BY g, {BIG} + n"),
        format!("SELECT COUNT(*) FILTER (WHERE {BIG} + n > 0) FROM v"),
        format!("SELECT g, COUNT(*) FILTER (WHERE {BIG} + n > 0) FROM v GROUP BY g"),
        format!("SELECT g, SUM({BIG} + n) FROM v GROUP BY ROLLUP(g)"),
        format!("SELECT g, COUNT(*) FILTER (WHERE {BIG} + n > 0) FROM v GROUP BY ROLLUP(g)"),
        format!("SELECT COUNT(*) FROM v GROUP BY ROLLUP({BIG} + n)"),
        format!("SELECT STRING_AGG(CAST(n AS TEXT), ',' ORDER BY {BIG} + n) FROM v"),
        format!("SELECT g, STRING_AGG(CAST(n AS TEXT), ',' ORDER BY {BIG} + n) FROM v GROUP BY g"),
        format!("SELECT SUM({BIG} + n) OVER () FROM v"),
        format!("SELECT g FROM v GROUP BY g HAVING MAX(n) + {BIG} > 0"),
        format!("SELECT MAX(n) + {BIG} FROM v"),
        format!("SELECT g, MAX(n) + {BIG} FROM v GROUP BY g"),
        format!("SELECT g, (SELECT MAX(n) FROM w WHERE w.id = v.g) + {BIG} FROM v GROUP BY g"),
        format!("SELECT v.id FROM v JOIN w ON {BIG} + v.n > w.n"),
        format!("SELECT v.id FROM v LEFT JOIN w ON {BIG} + v.n > w.n"),
        format!("SELECT v.id FROM v JOIN w ON w.id = v.id AND {BIG} + w.n > 0"),
        format!("SELECT v.id FROM v LEFT JOIN w ON w.id = v.id AND {BIG} + w.n > 0"),
        format!("SELECT v.id FROM v JOIN w ON w.n = v.n AND {BIG} + v.n > 0"),
        format!("SELECT v.id FROM v JOIN w ON w.n = v.n AND {BIG} + v.n > w.id"),
        format!("SELECT id FROM w WHERE n IN (SELECT {BIG} + n FROM v)"),
        format!("SELECT (SELECT {BIG} + n FROM v WHERE id = 1)"),
    ] {
        if let Ok(got) = rows(&db, &sql) {
            wrong.push(format!("{sql}: gave {got:?}, want an error"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_runtime_error_in_a_write_fails_it_and_changes_no_row() {
    let mut wrong = Vec::new();
    for (name, sql, check, want) in [
        (
            "delete",
            format!("DELETE FROM v WHERE {BIG} + n > 0"),
            "SELECT COUNT(*) FROM v",
            "Integer(2)",
        ),
        (
            "delete_one_row",
            format!("DELETE FROM v WHERE id = 2 OR {BIG} - 2 + n > 0"),
            "SELECT COUNT(*) FROM v",
            "Integer(2)",
        ),
        (
            "delete_correlated",
            format!(
                "DELETE FROM v WHERE EXISTS (SELECT 1 FROM w WHERE w.id = v.id) AND {BIG} + n > 0"
            ),
            "SELECT COUNT(*) FROM v",
            "Integer(2)",
        ),
        (
            "delete_in_list",
            format!("DELETE FROM v WHERE id IN (1, 2) AND {BIG} + n > 0"),
            "SELECT COUNT(*) FROM v",
            "Integer(2)",
        ),
        (
            "delete_in_select",
            format!("DELETE FROM v WHERE id IN (SELECT id FROM w) AND {BIG} + n > 0"),
            "SELECT COUNT(*) FROM v",
            "Integer(2)",
        ),
        (
            "upsert",
            format!("INSERT INTO w VALUES (1, 0) ON DUPLICATE KEY UPDATE n = n + {BIG}"),
            "SELECT n FROM w",
            "Integer(3)",
        ),
        (
            "update",
            format!("UPDATE v SET n = n + {BIG}"),
            "SELECT SUM(n) FROM v",
            "Integer(8)",
        ),
        (
            "update_correlated",
            format!("UPDATE v SET n = n + {BIG}, g = (SELECT MAX(n) FROM w WHERE w.id = v.id)"),
            "SELECT SUM(n) FROM v",
            "Integer(8)",
        ),
        (
            "update_correlated_where",
            format!(
                "UPDATE v SET g = 9 WHERE EXISTS (SELECT 1 FROM w WHERE w.id = v.id) AND {BIG} + n > 0"
            ),
            "SELECT SUM(g) FROM v",
            "Integer(2)",
        ),
        (
            "insert_select",
            format!("INSERT INTO w SELECT id + 10, {BIG} + n FROM v"),
            "SELECT COUNT(*) FROM w",
            "Integer(1)",
        ),
    ] {
        let db = open(name);
        if db.execute(&sql, ()).is_ok() {
            wrong.push(format!("{sql}: succeeded, want an error"));
        }
        match rows(&db, check) {
            Ok(got) if got == [want] => {}
            got => wrong.push(format!("{sql}: then {check} gave {got:?}, want {want}")),
        }
    }
    let db = open("default");
    db.execute(
        &format!(
            "CREATE TABLE d (id INTEGER PRIMARY KEY, x INTEGER DEFAULT ({BIG} + LENGTH('a')))"
        ),
        (),
    )
    .unwrap();
    if db.execute("INSERT INTO d (id) VALUES (1)", ()).is_ok() {
        wrong.push("INSERT with a failing DEFAULT: succeeded, want an error".to_string());
    }
    match rows(&db, "SELECT COUNT(*) FROM d") {
        Ok(got) if got == ["Integer(0)"] => {}
        got => wrong.push(format!("INSERT with a failing DEFAULT left {got:?}")),
    }
    db.execute(
        &format!("CREATE TABLE c (id INTEGER PRIMARY KEY, x INTEGER CHECK (x + {BIG} > 0))"),
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, -5)", ()).unwrap();
    if db
        .execute(
            "INSERT INTO c VALUES (1, 0) ON DUPLICATE KEY UPDATE x = 5",
            (),
        )
        .is_ok()
    {
        wrong.push("an upsert whose CHECK fails to evaluate: succeeded".to_string());
    }
    match rows(&db, "SELECT x FROM c") {
        Ok(got) if got == ["Integer(-5)"] => {}
        got => wrong.push(format!(
            "an upsert whose CHECK fails to evaluate left {got:?}"
        )),
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_null_input_stays_a_null_where_an_error_would_fail() {
    let db = open("nulls");
    let mut wrong = Vec::new();
    for (sql, want) in [
        ("SELECT z + n FROM v WHERE id = 1", vec!["NULL"]),
        (
            "SELECT id FROM v ORDER BY z + n, id",
            vec!["Integer(1)", "Integer(2)"],
        ),
        ("SELECT COUNT(*) FROM v GROUP BY z + n", vec!["Integer(2)"]),
        (
            "SELECT COUNT(*) FILTER (WHERE z + n > 0) FROM v",
            vec!["Integer(0)"],
        ),
        (
            "SELECT SUM(z + n) OVER () FROM v WHERE id = 1",
            vec!["NULL"],
        ),
        ("SELECT g FROM v GROUP BY g HAVING SUM(z) > 0", vec![]),
    ] {
        let want: Vec<String> = want.into_iter().map(String::from).collect();
        match rows(&db, sql) {
            Ok(got) if got == want => {}
            got => wrong.push(format!("{sql}: gave {got:?}, want {want:?}")),
        }
    }
    db.execute("DELETE FROM v WHERE z + n > 0", ()).unwrap();
    db.execute("UPDATE v SET n = z + n WHERE id = 2", ())
        .unwrap();
    match rows(&db, "SELECT id, n FROM v ORDER BY id") {
        Ok(got) if got == ["Integer(1)|Integer(3)", "Integer(2)|NULL"] => {}
        got => wrong.push(format!("DELETE and UPDATE over NULL left {got:?}")),
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_function_error_fails_the_statement_and_a_null_argument_gives_null() {
    let db = open("functions");
    let mut wrong = Vec::new();
    for sql in [
        "SELECT UPPER('a', 'b')",
        "SELECT UPPER('a', 'b') FROM v",
        "SELECT id FROM v WHERE UPPER('a', 'b') IS NULL",
        "SELECT DATE_TRUNC('fortnight', TIMESTAMP '2024-01-31 00:00:00')",
        "SELECT DATE_TRUNC('fortnight', TIMESTAMP '2024-01-31 00:00:00') FROM v",
        "SELECT SUBSTRING('abc', 'x')",
        "SELECT id FROM v ORDER BY DATE_TRUNC('fortnight', TIMESTAMP '2024-01-31 00:00:00')",
    ] {
        if let Ok(got) = rows(&db, sql) {
            wrong.push(format!("{sql}: gave {got:?}, want an error"));
        }
    }
    for (sql, want) in [
        ("SELECT SUBSTRING('abc', NULL)", "NULL"),
        ("SELECT SUBSTRING('abc', 1, NULL)", "NULL"),
        ("SELECT SUBSTR('abc', z) FROM v WHERE id = 1", "NULL"),
        ("SELECT JSON_EXTRACT('{\"a\": 1}', NULL)", "NULL"),
        ("SELECT MOD(5, 0)", "NULL"),
        ("SELECT MOD(n, 0) FROM v WHERE id = 1", "NULL"),
        ("SELECT 5 % 0", "NULL"),
        ("SELECT ABS(z) FROM v WHERE id = 1", "NULL"),
    ] {
        match rows(&db, sql) {
            Ok(got) if got == [want] => {}
            got => wrong.push(format!("{sql}: gave {got:?}, want {want}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
