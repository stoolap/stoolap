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

//! An error a row's condition raises is the statement's error, whichever
//! join path reads the row: hot and sealed, on the outer and the inner side

use stoolap::Database;

/// l: id 1..1000, k = id % 7 (unindexed), g = id but i64::MAX on row 500;
/// r: id 1..80, k = id % 7 (indexed), w = id % 5, v = id but i64::MAX on
/// row 40
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER, v INTEGER)",
        "CREATE INDEX r_k ON r (k)",
        "INSERT INTO l SELECT g.value, g.value % 7, \
         CASE WHEN g.value = 500 THEN 9223372036854775807 ELSE g.value END \
         FROM generate_series(1, 1000) g",
        "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5, \
         CASE WHEN g.value = 40 THEN 9223372036854775807 ELSE g.value END \
         FROM generate_series(1, 80) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

const BOTH: &[&str] = &["JOIN", "LEFT JOIN"];
const LEFT: &[&str] = &["LEFT JOIN"];
const NONE: &[&str] = &[];

fn answer(db: &Database, sql: &str) -> Result<usize, String> {
    db.query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map(|rows| rows.len())
        .map_err(|e| e.to_string())
}

#[test]
fn a_join_raises_the_error_a_side_condition_raises_for_a_row_it_returns() {
    // Each ON with the joins whose rows hold the failing row of l, and of
    // r, when the condition is in the WHERE and when it is in the ON: a
    // condition in the ON is asked only of the pairs the keys form
    let cases = [
        ("r.id = l.k", BOTH, NONE, BOTH, NONE),
        ("l.k = r.k", BOTH, BOTH, BOTH, BOTH),
        ("l.g = r.w", LEFT, NONE, NONE, NONE),
        ("l.k < r.k", BOTH, BOTH, BOTH, BOTH),
    ];
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
            fill(&db);
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            db
        } else {
            let db = Database::open("memory://join_side_error").unwrap();
            fill(&db);
            db
        };
        let mut wrong = Vec::new();
        let mut asked = 0;
        for (on, where_l, where_r, on_l, on_r) in cases {
            for join in ["JOIN", "LEFT JOIN"] {
                for (condition, in_where, in_on) in [
                    ("l.g + 1 > 0", where_l, on_l),
                    ("r.v + 1 > 0", where_r, on_r),
                ] {
                    for limit in ["", " LIMIT 100000"] {
                        for (place, reaches) in [("WHERE", in_where), ("AND", in_on)] {
                            if !reaches.contains(&join) {
                                continue;
                            }
                            let sql = format!(
                                "SELECT l.id, r.id FROM l {join} r ON {on} {place} {condition}{limit}"
                            );
                            asked += 1;
                            match answer(&db, &sql) {
                                Err(e) if e.contains("overflow") => {}
                                got => wrong.push(format!("sealed {sealed}: {sql}: {got:?}")),
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(asked, 42, "every case asked");
        assert!(
            wrong.is_empty(),
            "{} answered without the error:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }
}
