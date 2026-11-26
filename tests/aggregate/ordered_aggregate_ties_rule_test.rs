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

//! ARRAY_AGG, STRING_AGG and GROUP_CONCAT with ORDER BY order their rows by
//! the keys, then by the value (-0.0 before 0.0, INTEGER before FLOAT), then
//! by the row's separator; under DISTINCT each value stays at its first
//! place in that order. Rows answer alike however they were written, sealed
//! or read.

use stoolap::Database;

/// Group a ties on k = 1, with a NULL key, a NULL value and a value seen
/// again under another key; b holds two equal rows with other separators;
/// c only a NULL value
const ROWS: &str = "INSERT INTO t VALUES \
    (1, 'a', 1, 20, 0.0, 'y', '+', 0), (2, 'a', 1, 10, -0.0, 'x', '-', 0), \
    (3, 'a', 1, 30, 0.0, 'z', '*', 0), (4, 'a', NULL, 40, 1.5, 'w', '+', 0), \
    (5, 'a', 2, 10, NULL, 'x', '-', 0), (6, 'a', 1, NULL, NULL, NULL, '+', 0), \
    (7, 'b', 5, 50, 2.0, 'p', '+', 0), (8, 'b', 5, 50, 2.0, 'p', '*', 0), \
    (9, 'c', 1, NULL, NULL, NULL, '+', 0)";

/// t holds `ROWS` and u one row for groups a and b: hot, sealed, sealed
/// with rows 1, 2 and 7 rewritten, or rewritten and sealed again
fn open(name: &str, context: &str, dir: &tempfile::TempDir) -> Database {
    let dsn = match context {
        "hot" => format!("memory://{name}"),
        _ => format!("file://{}/{name}", dir.path().display()),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, k INTEGER, v INTEGER, \
         f FLOAT, s TEXT, sep TEXT, n INTEGER)",
        ROWS,
        "CREATE TABLE u (id INTEGER PRIMARY KEY, g TEXT, w INTEGER)",
        "INSERT INTO u VALUES (1, 'a', 7), (2, 'b', 7)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if context != "hot" {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    if context == "rewritten" || context == "resealed" {
        db.execute("UPDATE t SET n = 1 WHERE id IN (1, 2, 7)", ())
            .unwrap();
    }
    if context == "resealed" {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each row's values, one string per row
fn rows(result: stoolap::Result<stoolap::Rows>) -> Vec<String> {
    result
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).map_or("missing".into(), |v| v.to_string()))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[test]
fn ordered_aggregates_order_equal_keys_by_the_value() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for context in ["hot", "sealed", "rewritten", "resealed"] {
        let db = open(&format!("ordered_aggregate_ties_{context}"), context, &dir);
        for (sql, want) in [
            (
                "SELECT g, ARRAY_AGG(v ORDER BY k), ARRAY_AGG(v ORDER BY k DESC) \
                 FROM t GROUP BY g ORDER BY g",
                vec![
                    "a [10,20,30,10,40] [40,10,10,20,30]",
                    "b [50,50] [50,50]",
                    "c NULL NULL",
                ],
            ),
            (
                "SELECT g, STRING_AGG(s, '-' ORDER BY k), STRING_AGG(s, sep ORDER BY k), \
                 GROUP_CONCAT(s ORDER BY k NULLS FIRST) FROM t GROUP BY g ORDER BY g",
                vec![
                    "a x-y-z-x-w x+y*z-x+w w,x,y,z,x",
                    "b p-p p+p p,p",
                    "c NULL NULL NULL",
                ],
            ),
            (
                "SELECT g, ARRAY_AGG(DISTINCT v ORDER BY k), GROUP_CONCAT(DISTINCT s ORDER BY k), \
                 STRING_AGG(DISTINCT s, sep ORDER BY k) FROM t GROUP BY g ORDER BY g",
                vec![
                    "a [10,20,30,40] x,y,z,w x+y*z+w",
                    "b [50] p p+p",
                    "c NULL NULL NULL",
                ],
            ),
            (
                "SELECT ARRAY_AGG(f ORDER BY k), ARRAY_AGG(DISTINCT f ORDER BY k) \
                 FROM t WHERE g = 'a'",
                vec!["[-0,0,0,1.5] [-0,1.5]"],
            ),
            (
                "SELECT ARRAY_AGG(v ORDER BY k, s DESC) FROM t WHERE g = 'a'",
                vec!["[30,20,10,10,40]"],
            ),
            (
                "SELECT g, ARRAY_AGG(v ORDER BY k) FROM t WHERE id <> 2 GROUP BY g ORDER BY g",
                vec!["a [20,30,10,40]", "b [50,50]", "c NULL"],
            ),
            (
                "SELECT g FROM t GROUP BY g HAVING STRING_AGG(s, sep ORDER BY k) = 'x+y*z-x+w'",
                vec!["a"],
            ),
            (
                "SELECT g, CASE WHEN STRING_AGG(s, sep ORDER BY k) = 'p+p' THEN 'tie' \
                 ELSE 'other' END FROM t GROUP BY g ORDER BY g",
                vec!["a other", "b tie", "c other"],
            ),
            (
                "SELECT t.g, ARRAY_AGG(t.v ORDER BY u.w) FROM t JOIN u ON u.g = t.g \
                 GROUP BY t.g ORDER BY t.g",
                vec!["a [10,10,20,30,40]", "b [50,50]"],
            ),
            (
                "SELECT STRING_AGG(s, '' ORDER BY k) FROM (SELECT s, 1 AS k FROM t) d",
                vec!["ppwxxyz"],
            ),
            (
                "SELECT id, ARRAY_AGG(v ORDER BY k) OVER (PARTITION BY g) \
                 FROM t WHERE g = 'a' AND k = 1 ORDER BY id",
                vec![
                    "1 [10,20,30]",
                    "2 [10,20,30]",
                    "3 [10,20,30]",
                    "6 [10,20,30]",
                ],
            ),
        ] {
            let got = rows(db.query(sql, ()));
            if got != want {
                wrong.push(format!("{context} {sql}: {got:?}"));
            }
        }
        let got = rows(db.query(
            "SELECT STRING_AGG(s, sep ORDER BY k) FROM t WHERE g = $1",
            ("b",),
        ));
        if got != ["p+p"] {
            wrong.push(format!("{context} parameters: {got:?}"));
        }
        let prepared = db
            .prepare("SELECT ARRAY_AGG(DISTINCT v ORDER BY k) FROM t WHERE g = $1")
            .unwrap();
        for _ in 0..2 {
            let got = rows(prepared.query(("a",)));
            if got != ["[10,20,30,40]"] {
                wrong.push(format!("{context} prepared: {got:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn distinct_keeps_a_value_at_its_first_place_in_either_read_order() {
    let db = Database::open("memory://ordered_aggregate_ties_distinct").unwrap();
    let mut wrong = Vec::new();
    for (calls, rows_in, want) in [
        (
            "ARRAY_AGG(DISTINCT x ORDER BY k), STRING_AGG(DISTINCT x, '-' ORDER BY k)",
            [
                "SELECT 3 AS k, 5 AS x",
                "SELECT 1 AS k, 5 AS x",
                "SELECT 2 AS k, 7 AS x",
            ],
            "[5,7] 5-7",
        ),
        (
            "ARRAY_AGG(x ORDER BY k), STRING_AGG(x, '-' ORDER BY k), \
             STRING_AGG(DISTINCT x, '-' ORDER BY k)",
            [
                "SELECT 1 AS k, 1.0 AS x",
                "SELECT 1 AS k, 1 AS x",
                "SELECT 2 AS k, NULL AS x",
            ],
            "[1,1] 1-1 1",
        ),
        (
            "STRING_AGG(x, s ORDER BY k)",
            [
                "SELECT 1 AS k, 'b' AS x, '-' AS s",
                "SELECT 1 AS k, 'b' AS x, '*' AS s",
                "SELECT 0 AS k, 'a' AS x, '+' AS s",
            ],
            "a*b-b",
        ),
    ] {
        for order in [[0, 1, 2], [1, 0, 2]] {
            let source = order.map(|i| rows_in[i]).join(" UNION ALL ");
            let sql = format!("SELECT {calls} FROM ({source}) d");
            let got = rows(db.query(&sql, ()));
            if got != [want] {
                wrong.push(format!("{sql}: {got:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
