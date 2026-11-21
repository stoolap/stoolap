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

//! GROUPING(x, y, ...) reads one bit per argument, the first the most
//! significant, each the flag GROUPING of that argument alone reads,
//! wherever an expression over the groups stands, prepared or not

use stoolap::{Database, Rows, Value};

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, category TEXT, sub TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO products VALUES (1, 'A', 'x'), (2, 'A', 'y'), (3, 'B', 'x')",
        (),
    )
    .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
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

/// Each statement, its parameters and its rows in order
const ANSWERED: &[(&str, &[i64], &[&str])] = &[
    (
        "SELECT category, sub, GROUPING(category, sub), GROUPING(sub, category) FROM products \
         GROUP BY ROLLUP(category, sub) ORDER BY 3, 1, 2",
        &[],
        &[
            "A|x|0|0",
            "A|y|0|0",
            "B|x|0|0",
            "A|NULL|1|2",
            "B|NULL|1|2",
            "NULL|NULL|3|3",
        ],
    ),
    (
        "SELECT category, sub, COUNT(*) FROM products GROUP BY CUBE(category, sub) \
         HAVING GROUPING(category, sub) = 2 ORDER BY 2",
        &[],
        &["NULL|x|2", "NULL|y|1"],
    ),
    (
        "SELECT category, sub FROM products GROUP BY CUBE(category, sub) \
         ORDER BY GROUPING(sub, category) DESC, 1, 2",
        &[],
        &[
            "NULL|NULL",
            "A|NULL",
            "B|NULL",
            "NULL|x",
            "NULL|y",
            "A|x",
            "A|y",
            "B|x",
        ],
    ),
    (
        "SELECT category, sub, RANK() OVER (PARTITION BY GROUPING(category, sub) ORDER BY \
         COUNT(*) DESC) FROM products GROUP BY ROLLUP(category, sub) ORDER BY 3, 1, 2",
        &[],
        &[
            "A|x|1",
            "A|y|1",
            "A|NULL|1",
            "B|x|1",
            "NULL|NULL|1",
            "B|NULL|2",
        ],
    ),
    (
        "SELECT DISTINCT ON (GROUPING(category, sub)) category, sub FROM products \
         GROUP BY ROLLUP(category, sub) ORDER BY GROUPING(category, sub), category DESC, sub",
        &[],
        &["B|x", "B|NULL", "NULL|NULL"],
    ),
    (
        "SELECT category, sub, GROUPING(category, sub) * $1 FROM products \
         GROUP BY GROUPING SETS ((category), (sub)) ORDER BY 3, 1, 2",
        &[10],
        &["A|NULL|10", "B|NULL|10", "NULL|x|20", "NULL|y|20"],
    ),
    (
        "SELECT category, GROUPING(category, category, category) FROM products \
         GROUP BY ROLLUP(category) ORDER BY 2, 1",
        &[],
        &["A|0", "B|0", "NULL|7"],
    ),
    (
        "SELECT category, sub, GROUPING(category, sub) FROM products GROUP BY category, sub \
         ORDER BY 1, 2",
        &[],
        &["A|x|0", "A|y|0", "B|x|0"],
    ),
];

fn values(params: &[i64]) -> Vec<Value> {
    params.iter().map(|p| Value::Integer(*p)).collect()
}

#[test]
fn grouping_of_several_arguments_reads_one_bit_per_argument() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("grouping_several_arguments", &dir, sealed);
        for (sql, params, want) in ANSWERED {
            let want: Result<Vec<String>, String> =
                Ok(want.iter().map(|row| row.to_string()).collect());
            let statement = db.prepare(sql).unwrap();
            for run in 0..2 {
                let got = [
                    ("ad hoc", answer(db.query(sql, values(params)))),
                    ("prepared", answer(statement.query(values(params)))),
                ];
                for (how, got) in got {
                    if got != want {
                        wrong.push(format!("sealed {sealed}, {how}, run {run}: {sql}: {got:?}"));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn grouping_of_several_arguments_outside_a_grouped_statement_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = setup("grouping_several_arguments_ungrouped", &dir, false);
    assert_eq!(
        answer(db.query("SELECT GROUPING(category, sub) FROM products", ())),
        Err("Compile error: Function not found: GROUPING".to_string())
    );
}
