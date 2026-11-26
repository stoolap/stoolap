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

//! FIRST(x ORDER BY keys) and LAST(x ORDER BY keys) order the rows of a
//! non-NULL x by the keys, then by x ascending (-0.0 before 0.0), and take
//! the first and the last value: rows with equal keys answer alike however
//! they were written, sealed or read.

use stoolap::core::Value;
use stoolap::Database;

/// Group a ties on k = 1 with a NULL key and a NULL value beside, b has
/// only NULL keys, c only a NULL value
const ROWS: &str = "INSERT INTO t VALUES \
    (1, 'a', 1, 20, 0.0, 'y', 0), (2, 'a', 1, 10, -0.0, 'x', 0), \
    (3, 'a', 1, 30, 0.0, 'z', 0), (4, 'a', NULL, 40, 1.5, 'w', 0), \
    (5, 'a', 1, NULL, NULL, NULL, 0), (6, 'b', NULL, 60, 2.0, 'q', 0), \
    (7, 'b', NULL, 50, 2.0, 'p', 0), (8, 'c', 1, NULL, NULL, NULL, 0)";

/// t holds `ROWS` and u one row for groups a and b: hot, sealed, sealed
/// with rows 1 and 2 rewritten, or rewritten and sealed again
fn open(name: &str, context: &str, dir: &tempfile::TempDir) -> Database {
    let dsn = match context {
        "hot" => format!("memory://{name}"),
        _ => format!("file://{}/{name}", dir.path().display()),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, k INTEGER, v INTEGER, \
         f FLOAT, s TEXT, n INTEGER)",
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
        db.execute("UPDATE t SET n = 1 WHERE id IN (1, 2)", ())
            .unwrap();
    }
    if context == "resealed" {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each row's values, a FLOAT with its sign, one string per row
fn rows(result: stoolap::Result<stoolap::Rows>) -> Vec<String> {
    result
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| match row.get_value(i) {
                    Some(Value::Float(f)) => format!("{f:?}"),
                    Some(value) => format!("{value}"),
                    None => "missing".into(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[test]
fn first_and_last_order_equal_keys_by_the_value() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for context in ["hot", "sealed", "rewritten", "resealed"] {
        let db = open(&format!("first_last_ties_rule_{context}"), context, &dir);
        for (sql, want) in [
            (
                "SELECT g, FIRST(v ORDER BY k), LAST(v ORDER BY k), \
                 FIRST(v ORDER BY k DESC), LAST(v ORDER BY k DESC), \
                 FIRST(v ORDER BY k NULLS FIRST), LAST(v ORDER BY k NULLS FIRST) \
                 FROM t GROUP BY g ORDER BY g",
                vec![
                    "a 10 40 40 30 40 30",
                    "b 50 60 50 60 50 60",
                    "c NULL NULL NULL NULL NULL NULL",
                ],
            ),
            (
                "SELECT g, FIRST(f ORDER BY k), LAST(f ORDER BY k NULLS FIRST), \
                 FIRST(s ORDER BY k), LAST(s ORDER BY k NULLS FIRST), \
                 FIRST(v ORDER BY k, s DESC) FROM t GROUP BY g ORDER BY g",
                vec![
                    "a -0.0 0.0 x z 30",
                    "b 2.0 2.0 p q 60",
                    "c NULL NULL NULL NULL NULL",
                ],
            ),
            (
                "SELECT FIRST(v ORDER BY k), LAST(v ORDER BY k DESC) FROM t WHERE g = 'a'",
                vec!["10 30"],
            ),
            (
                "SELECT g, FIRST(v ORDER BY k) FROM t WHERE id <> 2 GROUP BY g ORDER BY g",
                vec!["a 20", "b 50", "c NULL"],
            ),
            (
                "SELECT g FROM t GROUP BY g HAVING FIRST(v ORDER BY k) = 10",
                vec!["a"],
            ),
            (
                "SELECT g, CASE WHEN LAST(v ORDER BY k DESC) = 30 THEN 'tie' ELSE 'other' END \
                 FROM t GROUP BY g ORDER BY g",
                vec!["a tie", "b other", "c other"],
            ),
            (
                "SELECT t.g, FIRST(t.v ORDER BY u.w), LAST(t.v ORDER BY u.w) \
                 FROM t JOIN u ON u.g = t.g GROUP BY t.g ORDER BY t.g",
                vec!["a 10 40", "b 50 60"],
            ),
            (
                "SELECT FIRST(v ORDER BY k), LAST(v ORDER BY k) \
                 FROM (SELECT v, 1 AS k FROM t WHERE g = 'a') d",
                vec!["10 40"],
            ),
            (
                "SELECT FIRST(x ORDER BY k), LAST(x ORDER BY k) \
                 FROM (SELECT 1 AS k, 1.0 AS x UNION ALL SELECT 1, 1) d",
                vec!["1 1.0"],
            ),
            (
                "SELECT id, FIRST(v ORDER BY k) OVER (PARTITION BY g), \
                 LAST(v ORDER BY k DESC) OVER (PARTITION BY g) \
                 FROM t WHERE g = 'a' AND k = 1 ORDER BY id",
                vec!["1 10 30", "2 10 30", "3 10 30", "5 10 30"],
            ),
        ] {
            let got = rows(db.query(sql, ()));
            if got != want {
                wrong.push(format!("{context} {sql}: {got:?}"));
            }
        }
        let got = rows(db.query(
            "SELECT FIRST(v ORDER BY k), LAST(s ORDER BY k NULLS FIRST) FROM t WHERE g = $1",
            ("a",),
        ));
        if got != ["10 z"] {
            wrong.push(format!("{context} parameters: {got:?}"));
        }
        let prepared = db
            .prepare("SELECT g, FIRST(f ORDER BY k) FROM t WHERE k = $1 GROUP BY g ORDER BY g")
            .unwrap();
        for _ in 0..2 {
            let got = rows(prepared.query((1,)));
            if got != ["a -0.0", "c NULL"] {
                wrong.push(format!("{context} prepared: {got:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
