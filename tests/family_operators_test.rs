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

#[test]
fn every_expression_form_answers_on_the_prepared_path_as_with_the_value_written() {
    let db = Database::open("memory://every_expression_form_answers_on_the_prepared_path").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, ts TIMESTAMP)",
        "INSERT INTO t VALUES (1, 1, '2024-01-01 00:00:00'), (2, 2, '2024-01-05 00:00:00'), \
         (3, NULL, '2024-01-09 00:00:00'), (4, 4, '2024-01-10 00:00:00')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
        "INSERT INTO u VALUES (1, '2024-01-02 00:00:00'), (2, '2024-01-10 00:00:00')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut failed = Vec::new();
    for form in [
        "ts + INTERVAL '1 day' > '2024-01-06 00:00:00'",
        "ts - INTERVAL '36 hours' < '2024-01-04 00:00:00'",
        "(k, id) = (1, 1)",
        "(k, id) < (2, 9)",
        "(k, id) IS NOT DISTINCT FROM (NULL, 3)",
        "(k, id) IN ((1, 1), (4, 4))",
        "(k, id) NOT IN ((1, id), (k, 9))",
    ] {
        for (shape, written) in [
            (
                format!("SELECT id FROM t WHERE {form} AND id > (SELECT MIN(id) FROM u)"),
                format!("SELECT id FROM t WHERE {form} AND id > 1"),
            ),
            (
                format!(
                    "SELECT id FROM t WHERE {form} AND EXISTS (SELECT 1 FROM u WHERE u.id = t.id)"
                ),
                format!("SELECT id FROM t WHERE {form} AND id IN (1, 2)"),
            ),
            (
                format!("SELECT id, ({form}) AS o, (SELECT MAX(id) FROM u) AS m FROM t"),
                format!("SELECT id, ({form}) AS o, 2 AS m FROM t"),
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
    for (shape, written) in [
        (
            "SELECT id FROM t WHERE ts > (SELECT MAX(ts) FROM u) - INTERVAL '2 days'",
            "SELECT id FROM t WHERE ts > TIMESTAMP '2024-01-10 00:00:00' - INTERVAL '2 days'",
        ),
        (
            "SELECT id FROM t WHERE ts - INTERVAL '1 day' < (SELECT ts FROM u WHERE u.id = t.id)",
            "SELECT id FROM t WHERE id IN (1, 2)",
        ),
        (
            "SELECT id FROM t WHERE (k, (SELECT MAX(id) FROM u)) = (2, 2)",
            "SELECT id FROM t WHERE (k, 2) = (2, 2)",
        ),
        (
            "SELECT t.id FROM t JOIN u ON (t.id, t.k) = (u.id, u.id) \
             AND t.k > (SELECT MIN(id) FROM u) - 1",
            "SELECT t.id FROM t JOIN u ON t.id = u.id AND t.k = u.id",
        ),
        (
            "SELECT id, (k, (SELECT u.id FROM u WHERE u.id = t.id)) = (1, 1) FROM t",
            "SELECT id, CASE id WHEN 1 THEN TRUE WHEN 3 THEN CAST(NULL AS BOOLEAN) ELSE FALSE END \
             FROM t",
        ),
    ] {
        let path = prepared_line(&db, shape);
        if path != "Prepared: yes" {
            failed.push(format!("{shape}: {path}"));
        }
        let (got, want) = (rows(&db, shape), rows(&db, written));
        if got != want {
            failed.push(format!("{shape}: {got:?}, written {want:?}"));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}
