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

//! `+ - * / %` and unary minus take INTEGER, FLOAT or NULL; any other
//! operand is an error naming its type, wherever the expression stands, and
//! a timestamp moves only by days or by text that reads as an interval

use stoolap::Database;

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, b BOOLEAN, n INTEGER, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, '42', TRUE, 5, '2024-01-01 00:00:00'), \
         (2, '8', FALSE, 7, '2024-01-02 00:00:00')",
        (),
    )
    .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The rows `sql` answers with `params`, or its error
fn answer(db: &Database, sql: &str, params: Vec<stoolap::Value>) -> Result<Vec<String>, String> {
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
fn an_operand_that_is_not_a_number_is_an_error_in_every_clause() {
    use stoolap::Value;
    let dir = tempfile::tempdir().unwrap();
    for sealed in [false, true] {
        let db = setup("arithmetic_operand_rule", &dir, sealed);
        let text = |sql: &str| (sql.to_string(), Vec::new(), "TEXT");
        let boolean = |sql: &str| (sql.to_string(), Vec::new(), "BOOLEAN");
        let mut cases = vec![
            text("SELECT '42' + 1"),
            text("SELECT 1 + '42'"),
            text("SELECT '42' * 2"),
            text("SELECT '42' + 1.5"),
            text("SELECT 'abc' - 1"),
            text("SELECT 7 % '2'"),
            text("SELECT 8 / '2'"),
            text("SELECT -'42'"),
            boolean("SELECT TRUE + 1"),
            boolean("SELECT -TRUE"),
            text("SELECT s + 1 FROM t"),
            text("SELECT -s FROM t"),
            text("SELECT id FROM t WHERE s + 1 = 43"),
            text("SELECT n, COUNT(*) FROM t GROUP BY n HAVING MAX(s) + 1 > 0"),
            text("SELECT CASE WHEN id > 0 THEN s * 2 END FROM t"),
            boolean("SELECT b + 1 FROM t"),
            boolean("SELECT id FROM t WHERE b * 2 = 2"),
            text("SELECT n + s FROM t ORDER BY id"),
        ];
        cases.push(("SELECT $1 + 1".to_string(), vec![Value::text("42")], "TEXT"));
        cases.push((
            "SELECT id FROM t WHERE n = $1 * 2".to_string(),
            vec![Value::text("3")],
            "TEXT",
        ));
        for (sql, params, got) in cases {
            let says = format!("arithmetic needs INTEGER or FLOAT, got {got}");
            let error = answer(&db, &sql, params.clone()).expect_err(&sql);
            assert!(error.contains(&says), "sealed {sealed}: {sql}: {error}");
            let prepared = db
                .prepare(&sql)
                .and_then(|stmt| stmt.query(params))
                .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
            let error = prepared.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(
                error.contains(&says),
                "sealed {sealed}: prepared {sql}: {error}"
            );
        }
        for sql in [
            "SELECT ts + 'abc' FROM t",
            "SELECT TIMESTAMP '2024-01-01 00:00:00' - 'abc'",
        ] {
            let error = answer(&db, sql, Vec::new()).expect_err(sql);
            assert!(error.contains("'abc'"), "sealed {sealed}: {sql}: {error}");
        }
        let error = db
            .execute("UPDATE t SET n = s + 1", ())
            .expect_err("UPDATE")
            .to_string();
        assert!(
            error.contains("arithmetic needs INTEGER or FLOAT, got TEXT"),
            "sealed {sealed}: {error}"
        );
        assert_eq!(
            answer(&db, "SELECT n FROM t ORDER BY id", Vec::new()).unwrap(),
            ["5", "7"],
            "sealed {sealed}: the UPDATE wrote nothing"
        );
        // Numbers, NULL and the timestamp forms answer as before
        for (sql, want) in [
            ("SELECT n + 1, n * 2.5, -n FROM t WHERE id = 1", "6|12.5|-5"),
            ("SELECT NULL + 1 IS NULL, -NULL IS NULL", "true|true"),
            ("SELECT CAST(s AS INTEGER) + 1 FROM t WHERE id = 1", "43"),
            (
                "SELECT ts + 1 = TIMESTAMP '2024-01-02 00:00:00' FROM t WHERE id = 1",
                "true",
            ),
            (
                "SELECT ts + '1 day' = TIMESTAMP '2024-01-02 00:00:00' FROM t WHERE id = 1",
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
