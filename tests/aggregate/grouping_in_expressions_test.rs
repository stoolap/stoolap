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

//! GROUPING(x) reads its group's flag wherever an expression over the groups
//! stands: inside SELECT expressions, in HAVING, in ORDER BY and in a window;
//! under a plain GROUP BY every group holds every item and it reads 0

use stoolap::{Database, Rows};

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

const OVER_GROUPING_SETS: &[(&str, &[&str])] = &[
    (
        "SELECT category, GROUPING(category) + 0 AS g FROM products GROUP BY ROLLUP(category) \
         ORDER BY 2, 1",
        &["A|0", "B|0", "NULL|1"],
    ),
    (
        "SELECT IIF(GROUPING(category) = 1, 'total', category) FROM products \
         GROUP BY ROLLUP(category) ORDER BY 1",
        &["A", "B", "total"],
    ),
    (
        "SELECT CASE WHEN GROUPING(sub) = 1 THEN '*' ELSE sub END, COUNT(*) FROM products \
         GROUP BY ROLLUP(category, sub) HAVING GROUPING(category) = 0 ORDER BY category, 1",
        &["*|2", "x|1", "y|1", "*|1", "x|1"],
    ),
    (
        "SELECT category, COUNT(*) FROM products GROUP BY ROLLUP(category) \
         HAVING GROUPING(category) = 0",
        &["A|2", "B|1"],
    ),
    (
        "SELECT category, COUNT(*) FROM products GROUP BY ROLLUP(category) \
         HAVING GROUPING(category) = 1 OR COUNT(*) = 1 ORDER BY COUNT(*)",
        &["B|1", "NULL|3"],
    ),
    (
        "SELECT category, COUNT(*) FROM products GROUP BY ROLLUP(category) \
         ORDER BY GROUPING(category) DESC, category",
        &["NULL|3", "A|2", "B|1"],
    ),
    (
        "SELECT category, COUNT(*) FROM products GROUP BY ROLLUP(category) \
         ORDER BY GROUPING(category) * -1, category LIMIT 2",
        &["NULL|3", "A|2"],
    ),
    (
        "SELECT category, GROUPING(category) AS g FROM products GROUP BY ROLLUP(category) \
         ORDER BY GROUPING(category) + 0 DESC, category",
        &["NULL|1", "A|0", "B|0"],
    ),
    (
        "SELECT category, sub, COUNT(*) FROM products GROUP BY CUBE(category, sub) \
         ORDER BY GROUPING(category) DESC, GROUPING(sub) DESC, category, sub",
        &[
            "NULL|NULL|3",
            "NULL|x|2",
            "NULL|y|1",
            "A|NULL|2",
            "B|NULL|1",
            "A|x|1",
            "A|y|1",
            "B|x|1",
        ],
    ),
    (
        "SELECT category, RANK() OVER (ORDER BY COUNT(*)) FROM products \
         GROUP BY ROLLUP(category) ORDER BY GROUPING(category) DESC, category",
        &["NULL|3", "A|2", "B|1"],
    ),
    (
        "SELECT category, SUM(COUNT(*)) OVER (PARTITION BY GROUPING(category) + 0) \
         FROM products GROUP BY ROLLUP(category) ORDER BY 1",
        &["A|3", "B|3", "NULL|3"],
    ),
    (
        "SELECT DISTINCT GROUPING(category) * 10 FROM products GROUP BY ROLLUP(category) \
         ORDER BY 1",
        &["0", "10"],
    ),
    (
        "SELECT category, GROUPING(category) FROM products \
         GROUP BY GROUPING SETS ((category), ()) ORDER BY GROUPING(category), category",
        &["A|0", "B|0", "NULL|1"],
    ),
];

const OVER_PLAIN_GROUPS: &[(&str, &[&str])] = &[
    (
        "SELECT category, GROUPING(category) + 1 FROM products GROUP BY category ORDER BY 1",
        &["A|1", "B|1"],
    ),
    (
        "SELECT category, COUNT(*) FROM products GROUP BY category \
         HAVING GROUPING(category) = 0 ORDER BY 1",
        &["A|2", "B|1"],
    ),
    (
        "SELECT category FROM products GROUP BY category \
         ORDER BY GROUPING(category), category DESC",
        &["B", "A"],
    ),
    (
        "SELECT category, RANK() OVER (PARTITION BY GROUPING(category) ORDER BY COUNT(*) DESC) \
         FROM products GROUP BY category ORDER BY 1",
        &["A|1", "B|2"],
    ),
];

#[test]
fn grouping_reads_the_flag_in_every_expression_over_the_groups() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("grouping_in_expressions", &dir, sealed);
        for (sql, want) in OVER_GROUPING_SETS.iter().chain(OVER_PLAIN_GROUPS) {
            let want: Vec<String> = want.iter().map(|row| row.to_string()).collect();
            assert_eq!(
                answer(db.query(sql, ())),
                Ok(want),
                "sealed {sealed}: {sql}"
            );
        }
    }
}

#[test]
fn grouping_in_a_prepared_statement_reads_the_flag() {
    let dir = tempfile::tempdir().unwrap();
    let db = setup("grouping_in_expressions_prepared", &dir, false);
    let statement = db
        .prepare(
            "SELECT IIF(GROUPING(category) = $1, 'total', category), COUNT(*) FROM products \
             GROUP BY ROLLUP(category) HAVING GROUPING(category) + $2 >= 1 \
             ORDER BY GROUPING(category) DESC, 1",
        )
        .unwrap();
    for (params, want) in [
        ((1, 1), vec!["total|3", "A|2", "B|1"]),
        ((1, 0), vec!["total|3"]),
    ] {
        let want: Vec<String> = want.into_iter().map(String::from).collect();
        assert_eq!(answer(statement.query(params)), Ok(want), "{params:?}");
    }
}

#[test]
fn grouping_outside_a_grouped_statement_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = setup("grouping_in_expressions_ungrouped", &dir, false);
    for sql in [
        "SELECT GROUPING(category) + 0 FROM products",
        "SELECT id FROM products ORDER BY GROUPING(category), id",
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY GROUPING(category) ORDER BY id) FROM products",
        "SELECT COUNT(*) FROM products HAVING GROUPING(category) = 0",
    ] {
        assert_eq!(
            answer(db.query(sql, ())),
            Err("Compile error: Function not found: GROUPING".to_string()),
            "{sql}"
        );
    }
}
