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

use stoolap::Database;

const MIN: &str = "(-9223372036854775807 - 1)";
const MAX: &str = "9223372036854775807";

/// `t (id, x, y)` with one row per (x, y) pair asked of it
fn table(name: &str, pairs: &[(String, String)]) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
        (),
    )
    .unwrap();
    for (id, (x, y)) in pairs.iter().enumerate() {
        db.execute(&format!("INSERT INTO t VALUES ({id}, {x}, {y})"), ())
            .unwrap();
    }
    db
}

/// The one value `sql` returns, as text, or its error
fn one(db: &Database, sql: &str) -> Result<String, String> {
    let mut rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let row = rows.next().ok_or("no row")?.map_err(|e| e.to_string())?;
    Ok(format!("{:?}", row.get_value(0)))
}

#[test]
fn test_every_operator_is_exact_or_an_error_in_every_position() {
    // (expression over x and y, x, y, the value or None for an error)
    let cases: &[(&str, String, String, Option<&str>)] = &[
        ("x + y", MAX.into(), "1".into(), None),
        ("x - y", MIN.into(), "1".into(), None),
        ("x * y", MAX.into(), "2".into(), None),
        ("x / y", MIN.into(), "-1".into(), None),
        ("x / y", "7".into(), "2".into(), Some("3")),
        ("x / y", "7".into(), "0".into(), Some("NULL")),
        ("x % y", MIN.into(), "-1".into(), Some("0")),
        ("x % y", "-7".into(), "2".into(), Some("-1")),
        ("x % y", "7".into(), "0".into(), Some("NULL")),
        ("MOD(x, y)", MIN.into(), "-1".into(), Some("0")),
        ("-x", MIN.into(), "0".into(), None),
        ("ABS(x)", MIN.into(), "0".into(), None),
        ("ABS(x)", "-5".into(), "0".into(), Some("5")),
        ("x << y", "1".into(), "63".into(), Some(MIN_TEXT)),
        ("x << y", "1".into(), "64".into(), None),
        ("x << y", "1".into(), "-1".into(), None),
        ("x >> y", "-8".into(), "1".into(), Some("-4")),
        ("x >> y", "1".into(), "64".into(), None),
    ];
    for (case, (expr, x, y, want)) in cases.iter().enumerate() {
        let db = table(&format!("integer_rule_{case}"), &[(x.clone(), y.clone())]);
        let literal = expr
            .replace('x', &format!("({x})"))
            .replace('y', &format!("({y})"));
        for sql in [
            format!("SELECT {literal}"),
            format!("SELECT {expr} FROM t"),
            format!("SELECT {literal} FROM t"),
            format!("SELECT id FROM t WHERE {expr} IS NOT NULL OR {expr} IS NULL"),
        ] {
            let got = one(&db, &sql);
            match want {
                None => assert!(got.is_err(), "{sql} with x = {x}, y = {y}: {got:?}"),
                Some("NULL") => assert!(
                    got.as_deref().is_ok_and(|v| v.contains("Null")) || sql.contains("WHERE"),
                    "{sql} with x = {x}, y = {y}: {got:?}"
                ),
                Some(value) if !sql.contains("WHERE") => assert_eq!(
                    got,
                    Ok(format!("Some(Integer({value}))")),
                    "{sql} with x = {x}, y = {y}"
                ),
                Some(_) => assert!(got.is_ok(), "{sql} with x = {x}, y = {y}: {got:?}"),
            }
        }
    }
}

const MIN_TEXT: &str = "-9223372036854775808";

#[test]
fn test_a_parameter_reads_the_same_rule() {
    let db = table("integer_rule_params", &[("1".into(), "1".into())]);
    let min = i64::MIN;
    for (sql, params, ok) in [
        ("SELECT $1 / $2", (min, -1i64), false),
        ("SELECT $1 % $2", (min, -1i64), true),
        ("SELECT -$1 + $2", (min, 0i64), false),
        ("SELECT $1 << $2", (1i64, 64i64), false),
    ] {
        let got = db
            .query(sql, params)
            .map_err(|e| e.to_string())
            .and_then(|mut rows| {
                rows.next()
                    .ok_or("no row".to_string())?
                    .map_err(|e| e.to_string())
                    .map(|_| ())
            });
        assert_eq!(got.is_ok(), ok, "{sql} {params:?}: {got:?}");
    }
}

#[test]
fn test_a_negated_literal_in_a_write_is_read_by_the_rule() {
    let db = table("integer_rule_writes", &[("1".into(), "1".into())]);
    for sql in [
        "UPDATE t SET x = - -9223372036854775808 WHERE id = 0",
        "INSERT INTO t VALUES (5, - -9223372036854775808, 0)",
    ] {
        assert!(db.execute(sql, ()).is_err(), "{sql}");
    }
    db.execute("UPDATE t SET x = -9223372036854775808 WHERE id = 0", ())
        .unwrap();
    for sql in ["SELECT -(-x) FROM t", "SELECT - - x FROM t"] {
        assert!(
            one(&db, sql).is_err(),
            "{sql}: -x overflows for x = i64::MIN"
        );
    }
}
