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

//! Each SELECT item answers the same values under the same names on every
//! path that projects a row: a streamed scan, a sorted one, a sort with a
//! LIMIT, a statement whose blocks the prepared runner reads, a view and a
//! CTE.

use stoolap::core::Value;
use stoolap::Database;

fn open() -> Database {
    let db = Database::open("memory://projection_paths").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b FLOAT, s TEXT, n TEXT)",
        "CREATE INDEX t_a ON t(a)",
        "INSERT INTO t SELECT value, value % 4, value * 1.5, 'text_' || CAST(value AS TEXT), \
         CASE WHEN value % 3 = 0 THEN NULL ELSE 'n' || CAST(value AS TEXT) END \
         FROM generate_series(1, 12)",
        "CREATE VIEW v AS SELECT * FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The column names and the rows' values `sql` answers with `params`, the
/// rows sorted
fn answer(
    db: &Database,
    sql: &str,
    params: &[Value],
) -> Result<(Vec<String>, Vec<String>), String> {
    let stmt = db.prepare(sql).map_err(|e| e.to_string())?;
    let rows = stmt.query(params.to_vec()).map_err(|e| e.to_string())?;
    let names = rows.columns().to_vec();
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(format!("{:?}", row.as_row().as_slice()));
    }
    out.sort();
    Ok((names, out))
}

#[test]
fn every_item_answers_alike_on_every_projecting_path() {
    let db = open();
    // Each item ends the list after `id`, which every path sorts by
    let items = [
        "a",
        "a AS x",
        "t.a",
        "*",
        "t.*",
        "1",
        "NULL",
        "'k'",
        "$1",
        "a + 1",
        "a + $1",
        "s || n",
        "s || $2",
        "COALESCE(n, s)",
        "COALESCE(n, NULL)",
        "COALESCE(n, $1)",
        "CASE WHEN a > 1 THEN s ELSE n END",
        "CASE WHEN a > 1 THEN s END",
        "CASE a WHEN 1 THEN 'one' END",
        "n || s",
        "s || 'x' || n",
        "s || a",
        "UPPER(s)",
        "b * 2",
        "a, a",
        "n IS NULL",
        "CAST(a AS TEXT)",
    ];
    let params = [Value::Integer(10), Value::text("_p")];
    let mut wrong = Vec::new();
    for item in items {
        let list = format!("id, {item}");
        let streamed = format!("SELECT {list} FROM t WHERE a = 1 OR a = 2");
        let paths = [
            ("sorted", format!("SELECT {list} FROM t WHERE a = 1 OR a = 2 ORDER BY id")),
            ("sorted with limit", format!("SELECT {list} FROM t WHERE a = 1 OR a = 2 ORDER BY id LIMIT 100")),
            ("runner", format!("SELECT {list} FROM t WHERE (a = 1 OR a = 2) AND 1 = (SELECT 1) ORDER BY id")),
            ("view", format!("SELECT {} FROM v AS t WHERE a = 1 OR a = 2 ORDER BY id", list)),
            ("cte", format!("WITH t AS (SELECT * FROM t) SELECT {list} FROM t WHERE a = 1 OR a = 2 ORDER BY id")),
        ];
        let reference = answer(&db, &streamed, &params);
        for (path, sql) in &paths {
            let got = answer(&db, sql, &params);
            if got != reference {
                wrong.push(format!(
                    "{item} on {path}\n  streamed {reference:?}\n  {path:8} {got:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
