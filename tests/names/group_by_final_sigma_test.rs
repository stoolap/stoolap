// Copyright 2026 Stoolap Contributors
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

//! A GROUP BY column whose name or qualifier ends in a capital sigma finds
//! the expressions that name it, as every other name does

use stoolap::Database;

const TABLES: [&str; 4] = [
    "CREATE TABLE u (id INTEGER PRIMARY KEY, \"ΑΣ\" INTEGER, v INTEGER)",
    "INSERT INTO u VALUES (1, 10, 1), (2, 20, 1), (3, 20, 2)",
    "CREATE TABLE w (id INTEGER PRIMARY KEY)",
    "INSERT INTO w VALUES (1)",
];

/// Each query with its rows, NULL as None, in the order its ORDER BY gives
const QUERIES: &[(&str, &[&[Option<i64>]])] = &[
    (
        "SELECT a.\"ΑΣ\", COUNT(*) FROM u AS a GROUP BY a.\"ΑΣ\" ORDER BY 1",
        &[&[Some(10), Some(1)], &[Some(20), Some(2)]],
    ),
    (
        "SELECT COUNT(*) AS n FROM u AS a CROSS JOIN w GROUP BY GROUPING SETS ((a.\"ΑΣ\")) ORDER BY 1",
        &[&[Some(1)], &[Some(2)]],
    ),
    (
        "SELECT \"ΑΣ\", SUM(v) FROM u CROSS JOIN w GROUP BY GROUPING SETS ((\"ΑΣ\")) ORDER BY 1",
        &[&[Some(10), Some(1)], &[Some(20), Some(3)]],
    ),
    (
        "SELECT \"ΚΑΣ\".\"ΑΣ\", COUNT(*) FROM u AS \"ΚΑΣ\" CROSS JOIN w \
         GROUP BY ROLLUP (\"ΚΑΣ\".\"ΑΣ\") ORDER BY 1 NULLS LAST",
        &[&[Some(10), Some(1)], &[Some(20), Some(2)], &[None, Some(3)]],
    ),
    (
        "SELECT a.\"ΑΣ\", COUNT(*) FROM u AS a CROSS JOIN w GROUP BY CUBE (a.\"ΑΣ\") ORDER BY 1 NULLS LAST",
        &[&[Some(10), Some(1)], &[Some(20), Some(2)], &[None, Some(3)]],
    ),
    (
        "SELECT \"ΑΣ\" AS \"ΤΑΣ\", COUNT(*) AS n FROM u GROUP BY \"ΤΑΣ\" HAVING \"ΤΑΣ\" > 15 ORDER BY 1",
        &[&[Some(20), Some(2)]],
    ),
];

fn rows(rows: stoolap::Rows) -> Vec<Vec<Option<i64>>> {
    rows.map(|row| {
        let row = row.unwrap();
        (0..row.len())
            .map(|i| row.get::<Option<i64>>(i).unwrap())
            .collect()
    })
    .collect()
}

#[test]
fn a_group_by_column_ending_in_a_capital_sigma_groups_its_rows() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = match sealed {
            true => format!("file://{}?sync_mode=none", dir.path().display()),
            false => "memory://group_by_final_sigma".to_string(),
        };
        let db = Database::open(&dsn).unwrap();
        for sql in TABLES {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (sql, expected) in QUERIES {
            let expected: Vec<Vec<Option<i64>>> = expected.iter().map(|row| row.to_vec()).collect();
            for prepared in [false, true] {
                let got = match prepared {
                    true => db.prepare(sql).and_then(|stmt| stmt.query(())).map(rows),
                    false => db.query(sql, ()).map(rows),
                };
                if got.as_ref().ok() != Some(&expected) {
                    wrong.push(format!(
                        "{sql} (sealed {sealed}, prepared {prepared}): {got:?}, not {expected:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
