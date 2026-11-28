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

//! A CAST of a value its target type cannot read is an error naming both,
//! wherever it stands; NULL stays NULL and a value that converts converts

use stoolap::{Database, Value};

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 'abc', '2024-01-01 00:00:00'), (2, '42', NULL)",
        (),
    )
    .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The rows `sql` answers with `params`, or its error
fn answer(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<String>, String> {
    db.query(sql, params)
        .map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

#[test]
fn a_cast_its_target_cannot_read_is_an_error_in_every_clause() {
    let dir = tempfile::tempdir().unwrap();
    for sealed in [false, true] {
        let db = setup("cast_unconvertible", &dir, sealed);
        let none = Vec::new;
        let mut cases: Vec<(String, Vec<Value>, &str)> = [
            ("SELECT CAST('abc' AS INTEGER)", "INTEGER"),
            ("SELECT CAST('4.7' AS INTEGER)", "INTEGER"),
            ("SELECT CAST('1e3' AS INTEGER)", "INTEGER"),
            ("SELECT CAST('abc' AS FLOAT)", "FLOAT"),
            ("SELECT CAST('abc' AS TIMESTAMP)", "TIMESTAMP"),
            ("SELECT CAST('2024-13-45' AS TIMESTAMP)", "TIMESTAMP"),
            ("SELECT CAST('abc' AS BOOLEAN)", "BOOLEAN"),
            ("SELECT CAST('abc' AS JSON)", "JSON"),
            ("SELECT CAST('abc' AS VECTOR)", "VECTOR"),
            // A DATE reads as a TIMESTAMP does
            ("SELECT CAST('abc' AS DATE)", "TIMESTAMP"),
            ("SELECT CAST(TRUE AS DATE)", "TIMESTAMP"),
            ("SELECT CAST(s AS DATE) FROM t", "TIMESTAMP"),
            (
                "SELECT CAST(TIMESTAMP '2024-01-01 00:00:00' AS INTEGER)",
                "INTEGER",
            ),
            ("SELECT CAST(s AS INTEGER) FROM t", "INTEGER"),
            ("SELECT id FROM t WHERE CAST(s AS INTEGER) = 42", "INTEGER"),
            ("SELECT CAST(ts AS INTEGER) FROM t WHERE id = 1", "INTEGER"),
            (
                "SELECT CASE WHEN id > 0 THEN CAST(s AS FLOAT) END FROM t",
                "FLOAT",
            ),
        ]
        .into_iter()
        .map(|(sql, target)| (sql.to_string(), none(), target))
        .collect();
        cases.push((
            "SELECT CAST($1 AS INTEGER)".to_string(),
            vec![Value::text("abc")],
            "INTEGER",
        ));
        for (sql, params, target) in cases {
            let error = answer(&db, &sql, params.clone()).expect_err(&sql);
            let says = "cannot cast";
            assert!(
                error.contains(says) && error.contains(&format!("to {target}")),
                "sealed {sealed}: {sql}: {error}"
            );
            let prepared = db
                .prepare(&sql)
                .and_then(|stmt| stmt.query(params))
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                prepared.contains(says),
                "sealed {sealed}: prepared {sql}: {prepared}"
            );
        }
        // NULL stays NULL, and what converts still converts
        for (sql, want) in [
            ("SELECT CAST(NULL AS INTEGER) IS NULL", "true"),
            (
                "SELECT CAST(ts AS INTEGER) IS NULL FROM t WHERE id = 2",
                "true",
            ),
            (
                "SELECT CAST('42' AS INTEGER) + CAST(' 8 ' AS INTEGER)",
                "50",
            ),
            ("SELECT CAST('1.5' AS FLOAT)", "1.5"),
            ("SELECT CAST('yes' AS BOOLEAN)", "true"),
            ("SELECT CAST(s AS INTEGER) FROM t WHERE id = 2", "42"),
            (
                "SELECT CAST('2024-01-15 14:30:00' AS DATE) = TIMESTAMP '2024-01-15 00:00:00'",
                "true",
            ),
            (
                "SELECT CAST(ts AS DATE) IS NULL FROM t WHERE id = 2",
                "true",
            ),
            ("SELECT CAST(NULL AS DATE) IS NULL", "true"),
            // An INTEGER is nanoseconds since the epoch, before it too
            (
                "SELECT CAST(-1 AS TIMESTAMP) = TIMESTAMP '1969-12-31 23:59:59.999999999'",
                "true",
            ),
            (
                "SELECT CAST(-1500000000 AS TIMESTAMP) = TIMESTAMP '1969-12-31 23:59:58.5'",
                "true",
            ),
            (
                "SELECT CAST(-1 AS DATE) = TIMESTAMP '1969-12-31 00:00:00'",
                "true",
            ),
            (
                "SELECT CAST(1500000000 AS DATE) = TIMESTAMP '1970-01-01 00:00:00'",
                "true",
            ),
        ] {
            assert_eq!(
                answer(&db, sql, Vec::new()),
                Ok(vec![want.to_string()]),
                "sealed {sealed}: {sql}"
            );
        }
        db.close().unwrap();
    }
}
