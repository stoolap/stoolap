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

//! Every operator keeps a statement with a subquery prepared, and answers
//! there as it does with the subquery's value written in its place

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(format!(
            "{:?}",
            (0..row.len())
                .map(|i| row.get::<stoolap::Value>(i).unwrap())
                .collect::<Vec<_>>()
        ));
    }
    out.sort_unstable();
    Ok(out)
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.contains("Prepared"))
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[test]
fn every_operator_answers_on_the_prepared_path_as_with_the_value_written() {
    let db = Database::open("memory://every_operator_answers_on_the_prepared_path").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, b BOOLEAN, j JSON, \
         e VECTOR(2))",
        "INSERT INTO t VALUES (1, 1, 'Ab', true, '{\"a\":1}', '[1.0, 0.0]'), \
         (2, 2, 'ab', false, '{\"a\":2}', '[0.0, 1.0]'), \
         (3, 3, 'cd', NULL, '[5, 6]', '[1.0, 1.0]'), (4, 4, 'Bx', true, NULL, NULL)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, g INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut failed = Vec::new();
    for operator in [
        "s || 'x' <> 'zx'",
        "(k > 1) XOR (id > 2)",
        "s ILIKE 'a%'",
        "s NOT ILIKE 'a%'",
        "s GLOB 'a*'",
        "s NOT GLOB 'a*'",
        "s REGEXP '^[ab]'",
        "s NOT REGEXP '^[ab]'",
        "b IS TRUE",
        "b IS NOT FALSE",
        "j -> 'a' IS NOT NULL",
        "j ->> 'a' = '1'",
        "j[0] = 5",
        "e <=> '[1.0, 0.0]' < 0.5",
        "(k & 1) = 1",
        "(k | 4) = 5",
        "(k ^ 1) = 0",
        "(k << 1) = 4",
        "(k >> 1) = 1",
        "+k = 1",
        "~k = -2",
    ] {
        for (shape, written) in [
            (
                format!("SELECT id FROM t WHERE {operator} AND id > (SELECT MIN(id) FROM u)"),
                format!("SELECT id FROM t WHERE {operator} AND id > 1"),
            ),
            (
                format!(
                    "SELECT id FROM t WHERE {operator} \
                     AND EXISTS (SELECT 1 FROM u WHERE u.id = t.id)"
                ),
                format!("SELECT id FROM t WHERE {operator} AND id IN (1, 2)"),
            ),
            (
                format!("SELECT id FROM t WHERE {operator} AND k IN (SELECT t.k FROM u)"),
                format!("SELECT id FROM t WHERE {operator}"),
            ),
            (
                format!("SELECT id, ({operator}) AS o, (SELECT MAX(id) FROM u) AS m FROM t"),
                format!("SELECT id, ({operator}) AS o, 2 AS m FROM t"),
            ),
        ] {
            let path = prepared_line(&db, &shape);
            if path != "Prepared: yes" {
                failed.push(format!("{shape}: {path}"));
            }
            let (got, want) = (rows(&db, &shape), rows(&db, &written));
            if got != want {
                failed.push(format!("{shape}: {got:?}, written {want:?}"));
            }
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}
