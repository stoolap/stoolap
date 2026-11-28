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

//! A statement over one table naming a column the table does not have, or
//! a table no table or view has, answers that error in every clause, hot
//! and sealed, prepared or not, and compiles nothing to find it

use stoolap::{Database, Rows, Value};

const COLUMN: &str = "column 'missing' not found";
const TABLE: &str = "table or view 'nope' not found";
/// A GROUP BY key or an aggregate's separator no column has, quoted or not
const COMPILED: &str = "Compile error: Column 'missing' not found";

/// Each statement, the parameters it takes and its error
const CASES: &[(&str, &[i64], &str)] = &[
    ("SELECT missing FROM t", &[], COLUMN),
    ("SELECT id, missing + 1 FROM t", &[], COLUMN),
    (
        "SELECT CASE WHEN missing > 1 THEN 1 END FROM t",
        &[],
        COLUMN,
    ),
    ("SELECT id FROM t WHERE missing = 1", &[], COLUMN),
    ("SELECT id FROM t WHERE missing = $1", &[1], COLUMN),
    ("SELECT missing FROM t WHERE id = $1", &[1], COLUMN),
    ("SELECT missing FROM t WHERE 1 = 0", &[], COLUMN),
    (
        "SELECT id FROM t WHERE missing = $1 AND 1 = 0",
        &[1],
        COLUMN,
    ),
    ("SELECT id FROM t ORDER BY missing", &[], COLUMN),
    ("SELECT id FROM t ORDER BY missing LIMIT $1", &[1], COLUMN),
    (
        "SELECT COUNT(*) FROM t WHERE missing IS NOT NULL",
        &[],
        COLUMN,
    ),
    (
        "SELECT g, COUNT(*) FROM t WHERE missing > $1 GROUP BY g",
        &[1],
        COLUMN,
    ),
    ("SELECT g, SUM(missing) FROM t GROUP BY g", &[], COLUMN),
    ("SELECT SUM(missing) FROM t", &[], COLUMN),
    ("SELECT COUNT(missing) FROM t WHERE id > $1", &[0], COLUMN),
    ("SELECT AVG(DISTINCT missing) FROM t", &[], COLUMN),
    (
        "SELECT g, MAX(missing) FROM t GROUP BY g ORDER BY g",
        &[],
        COLUMN,
    ),
    (
        "SELECT g FROM t GROUP BY g HAVING SUM(missing) > $1",
        &[0],
        COLUMN,
    ),
    ("SELECT DISTINCT missing FROM t", &[], COLUMN),
    // A double-quoted name is a name too
    ("SELECT \"missing\" FROM t", &[], COLUMN),
    ("SELECT \"missing\"", &[], COLUMN),
    ("SELECT id, \"missing\" || 'x' FROM t", &[], COLUMN),
    (
        "SELECT CASE WHEN a > 1 THEN \"missing\" END FROM t",
        &[],
        COLUMN,
    ),
    ("SELECT id FROM t WHERE a = \"missing\"", &[], COLUMN),
    ("SELECT id FROM t WHERE a IN (\"missing\", 1)", &[], COLUMN),
    ("SELECT id FROM t WHERE a LIKE \"missing\"", &[], COLUMN),
    (
        "SELECT id FROM t WHERE id > $1 ORDER BY \"missing\"",
        &[0],
        COLUMN,
    ),
    ("SELECT COUNT(*) FROM t GROUP BY missing", &[], COMPILED),
    ("SELECT COUNT(*) FROM t GROUP BY \"missing\"", &[], COMPILED),
    (
        "SELECT g FROM t GROUP BY g HAVING MAX(\"missing\") > $1",
        &[0],
        COLUMN,
    ),
    ("SELECT COUNT(\"missing\") FROM t", &[], COLUMN),
    ("SELECT MAX(\"missing\") FROM t WHERE id > $1", &[1], COLUMN),
    (
        "SELECT g, MIN(\"missing\") FROM t GROUP BY g ORDER BY g",
        &[],
        COLUMN,
    ),
    ("SELECT STRING_AGG(\"missing\", ',') FROM t", &[], COLUMN),
    (
        "SELECT g, STRING_AGG(CAST(a AS TEXT), missing) FROM t GROUP BY g",
        &[],
        COMPILED,
    ),
    (
        "SELECT g, STRING_AGG(CAST(a AS TEXT), \"missing\") FROM t GROUP BY g",
        &[],
        COMPILED,
    ),
    (
        "SELECT COUNT(DISTINCT \"missing\"), SUM(a) FROM t",
        &[],
        COLUMN,
    ),
    (
        "SELECT id FROM t WHERE a IN (SELECT \"missing\" FROM t)",
        &[],
        COLUMN,
    ),
    ("SELECT * FROM nope", &[], TABLE),
    ("SELECT COUNT(*) FROM nope", &[], TABLE),
    ("SELECT id FROM nope WHERE id = $1", &[1], TABLE),
];

fn open(dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 1), (2, 20, 1), (3, 30, 2)",
        (),
    )
    .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn values(params: &[i64]) -> Vec<Value> {
    params.iter().map(|p| Value::Integer(*p)).collect()
}

fn error(rows: stoolap::Result<Rows>) -> Option<String> {
    match rows {
        Ok(mut rows) => rows.find_map(|row| row.err()),
        Err(e) => Some(e),
    }
    .map(|e| e.to_string())
}

fn prepared_error(db: &Database, sql: &str, params: &[i64]) -> Option<String> {
    match db.prepare(sql) {
        Ok(stmt) => error(stmt.query(values(params))),
        Err(e) => Some(e.to_string()),
    }
}

#[test]
fn a_name_the_table_does_not_have_is_its_error_hot_and_sealed() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        for (sql, params, want) in CASES {
            for run in 0..2 {
                let context = format!("sealed {sealed}, run {run}: {sql}");
                let got = error(db.query(sql, values(params)));
                assert_eq!(got.as_deref(), Some(*want), "{context}");
                let got = prepared_error(&db, sql, params);
                assert_eq!(got.as_deref(), Some(*want), "prepared, {context}");
            }
        }
    }
}

fn answer(rows: stoolap::Result<Rows>) -> Result<Vec<String>, String> {
    rows.map_err(|e| e.to_string())?
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
fn a_quoted_name_reads_the_column_it_names_and_writes_nothing_else() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed);
        let cases: &[(&str, &[&str])] = &[
            (
                "SELECT \"a\" FROM t WHERE \"g\" = 1 ORDER BY id",
                &["10", "20"],
            ),
            (
                "SELECT id FROM t WHERE a IN (\"a\") ORDER BY id",
                &["1", "2", "3"],
            ),
            ("SELECT MAX(\"a\") FROM t", &["30"]),
            (
                "SELECT g, COUNT(*) FROM t GROUP BY \"g\" ORDER BY g",
                &["1|2", "2|1"],
            ),
        ];
        for (sql, want) in cases {
            let got = answer(db.query(sql, ()));
            let want = want.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            assert_eq!(got, Ok(want), "sealed {sealed}: {sql}");
        }
        let got = db.execute("UPDATE t SET a = \"missing\"", ());
        assert_eq!(
            got.map_err(|e| e.to_string()),
            Err(COLUMN.to_string()),
            "sealed {sealed}: UPDATE"
        );
        let got = answer(db.query("SELECT a FROM t ORDER BY id", ()));
        assert_eq!(
            got,
            Ok(vec!["10".into(), "20".into(), "30".into()]),
            "sealed {sealed}"
        );
    }
}
