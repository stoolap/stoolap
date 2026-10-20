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

//! A qualified star expands to its source's columns, a grouped item is
//! computed from its expression, a bare GROUP BY name binds the input
//! column first, and a grouped statement runs its windows over the groups,
//! whatever the source

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, k INTEGER)",
        "INSERT INTO t VALUES (1, 1, 10), (2, 1, 20), (3, 2, 30)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, z INTEGER)",
        "INSERT INTO s VALUES (1, 7), (2, 8), (3, 9)",
        "CREATE VIEW v AS SELECT id, g, k FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

type Answer = Result<(Vec<String>, Vec<Vec<String>>), String>;

/// A shape: its name, SELECT list, GROUP BY, ORDER BY, the columns when
/// checked, and the rows
type Shape = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    Option<Vec<&'static str>>,
    Vec<Vec<String>>,
);

fn answer(db: &Database, sql: &str, params: &[i64]) -> Answer {
    let params: Vec<stoolap::Value> = params.iter().map(|&p| stoolap::Value::Integer(p)).collect();
    let rows = if params.is_empty() {
        db.query(sql, ())
    } else {
        db.prepare(sql)
            .and_then(|statement| statement.query(params))
    }
    .map_err(|e| e.to_string())?;
    let columns = rows.columns().to_vec();
    let width = columns.len();
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..width)
                .map(|i| match row.get_value(i) {
                    Some(stoolap::Value::Integer(n)) => n.to_string(),
                    Some(value) if value.is_null() => "NULL".to_string(),
                    Some(value) => format!("{value:?}"),
                    None => "missing".to_string(),
                })
                .collect(),
        );
    }
    Ok((columns, out))
}

fn rows(values: &[&[i64]]) -> Vec<Vec<String>> {
    values
        .iter()
        .map(|row| row.iter().map(|v| v.to_string()).collect())
        .collect()
}

#[test]
fn every_source_projects_by_one_rule() {
    let db = open("projection_rules_matrix");
    // (name, WITH, FROM, WHERE)
    let sources = [
        ("table", "", "t AS src", ""),
        ("join", "", "t AS src JOIN s ON s.id = src.id", ""),
        ("view", "", "v AS src", ""),
        ("derived", "", "(SELECT id, g, k FROM t) AS src", ""),
        ("cte", "WITH c AS (SELECT id, g, k FROM t) ", "c AS src", ""),
        (
            "cte_columns",
            "WITH c(id, g, k) AS (SELECT id, g, k FROM t) ",
            "c AS src",
            "",
        ),
        (
            "cte_materialized",
            "WITH c AS (SELECT id, g, k FROM t) ",
            "c AS src",
            "src.id IN (SELECT id FROM c)",
        ),
        (
            "values",
            "",
            "(VALUES (1, 1, 10), (2, 1, 20), (3, 2, 30)) AS src(id, g, k)",
            "",
        ),
        (
            "temporal",
            "",
            "t AS OF TIMESTAMP '2050-01-01 00:00:00' AS src",
            "",
        ),
    ];
    let star = ["id", "g", "k"];
    let shapes: Vec<Shape> = vec![
        (
            "star beside an alias",
            "src.*, -src.id AS x",
            "",
            "src.id",
            Some([&star[..], &["x"]].concat()),
            rows(&[&[1, 1, 10, -1], &[2, 1, 20, -2], &[3, 2, 30, -3]]),
        ),
        (
            "star alone",
            "src.*",
            "",
            "1",
            Some(star.to_vec()),
            rows(&[&[1, 1, 10], &[2, 1, 20], &[3, 2, 30]]),
        ),
        (
            "star beside a window",
            "src.*, ROW_NUMBER() OVER (ORDER BY src.id) AS rn",
            "",
            "src.id",
            Some([&star[..], &["rn"]].concat()),
            rows(&[&[1, 1, 10, 1], &[2, 1, 20, 2], &[3, 2, 30, 3]]),
        ),
        (
            "grouped alias named as a grouping column",
            "src.g, -src.k AS k",
            "src.g, src.k",
            "1, 2",
            None,
            rows(&[&[1, -20], &[1, -10], &[2, -30]]),
        ),
        (
            "grouped expression named as a grouping column",
            "src.g, src.k + 1 AS k",
            "src.g, src.k",
            "1, 2",
            None,
            rows(&[&[1, 11], &[1, 21], &[2, 31]]),
        ),
        (
            "grouped call named as a grouping column",
            "src.g, ABS(src.k - 100) AS k",
            "src.g, src.k",
            "1, 2",
            None,
            rows(&[&[1, 80], &[1, 90], &[2, 70]]),
        ),
        (
            "grouped alias over a fresh name",
            "src.g, -src.k AS x",
            "src.g, src.k",
            "1, 2",
            None,
            rows(&[&[1, -20], &[1, -10], &[2, -30]]),
        ),
        (
            "window beside a call named as an input column",
            "ABS(src.g - 5) AS g, ROW_NUMBER() OVER (ORDER BY src.id) AS rn",
            "",
            "2",
            None,
            rows(&[&[4, 1], &[4, 2], &[3, 3]]),
        ),
        (
            "window over groups beside a call named as a grouping column",
            "ABS(src.g - 5) AS g, ROW_NUMBER() OVER (ORDER BY src.g) AS rn",
            "src.g",
            "2",
            None,
            rows(&[&[4, 1], &[3, 2]]),
        ),
        (
            "aggregate and window over groups",
            "src.g, COUNT(*) AS n, ROW_NUMBER() OVER (ORDER BY src.g) AS rn",
            "src.g",
            "1",
            None,
            rows(&[&[1, 2, 1], &[2, 1, 2]]),
        ),
        (
            "window over groups",
            "src.g, ROW_NUMBER() OVER (ORDER BY src.g) AS rn",
            "src.g",
            "1",
            None,
            rows(&[&[1, 1], &[2, 2]]),
        ),
        (
            "window over an aggregate",
            "src.g, SUM(COUNT(*)) OVER () AS total",
            "src.g",
            "1",
            None,
            rows(&[&[1, 3], &[2, 3]]),
        ),
        (
            "GROUP BY name an aggregate alias shadows",
            "MIN(src.k) AS g, COUNT(*) AS n",
            "g",
            "1",
            None,
            rows(&[&[10, 2], &[30, 1]]),
        ),
        (
            "GROUP BY name an expression alias carries",
            "src.g * 0 AS g, COUNT(*) AS n",
            "g",
            "2",
            None,
            rows(&[&[0, 3]]),
        ),
    ];
    // (name, LIMIT, prepared)
    let pipelines = [
        ("plain", "", false),
        ("limit", " LIMIT 10", false),
        ("prepared", "", true),
    ];
    let mut wrong = Vec::new();
    let mut ran = 0;
    for (source, with, from, filter) in sources {
        for (shape, items, group_by, order_by, columns, want) in &shapes {
            for (pipeline, limit, prepared) in pipelines {
                let mut conditions: Vec<&str> = Vec::new();
                if !filter.is_empty() {
                    conditions.push(filter);
                }
                if prepared {
                    conditions.push("src.id > $1");
                }
                let sql = format!(
                    "{with}SELECT {items} FROM {from}{}{} ORDER BY {order_by}{limit}",
                    if conditions.is_empty() {
                        String::new()
                    } else {
                        format!(" WHERE {}", conditions.join(" AND "))
                    },
                    if group_by.is_empty() {
                        String::new()
                    } else {
                        format!(" GROUP BY {group_by}")
                    },
                );
                let params: &[i64] = if prepared { &[0] } else { &[] };
                let got = answer(&db, &sql, params);
                ran += 1;
                let right = match &got {
                    Ok((got_columns, got_rows)) => {
                        got_rows == want
                            && columns
                                .as_ref()
                                .is_none_or(|columns| got_columns == columns)
                    }
                    Err(_) => false,
                };
                if !right {
                    wrong.push(format!(
                        "[{source}/{pipeline}] {shape}: {sql}\n  got  {got:?}\n  want {columns:?} {want:?}"
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {ran} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_source_without_a_table_projects_by_the_same_rule() {
    let db = open("projection_rules_no_table");
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER, \"Κ\" INTEGER)",
        "INSERT INTO u VALUES (1, 1, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, columns, want) in [
        (
            "SELECT g.* FROM generate_series(1, 2) AS g ORDER BY 1",
            vec!["value"],
            rows(&[&[1], &[2]]),
        ),
        (
            "SELECT g.*, -value AS x FROM generate_series(1, 2) AS g ORDER BY 1",
            vec!["value", "x"],
            rows(&[&[1, -1], &[2, -2]]),
        ),
        (
            "SELECT value % 2 AS p, COUNT(*) AS n, ROW_NUMBER() OVER (ORDER BY value % 2) AS rn \
             FROM generate_series(1, 3) GROUP BY value % 2 ORDER BY 1",
            vec!["p", "n", "rn"],
            rows(&[&[0, 1, 1], &[1, 2, 2]]),
        ),
        (
            "SELECT COUNT(*) AS n, ROW_NUMBER() OVER () AS rn",
            vec!["n", "rn"],
            rows(&[&[1, 1]]),
        ),
        (
            "SELECT u.a AS \"Κ\", COUNT(*) AS n FROM u GROUP BY u.a, u.\"Κ\" HAVING \"Κ\" > 5",
            vec!["Κ", "n"],
            rows(&[&[1, 1]]),
        ),
        (
            "SELECT \"ΟΣ\".* FROM t AS \"ΟΣ\" JOIN s ON s.id = \"ΟΣ\".id ORDER BY 1",
            vec!["id", "g", "k"],
            rows(&[&[1, 1, 10], &[2, 1, 20], &[3, 2, 30]]),
        ),
    ] {
        let got = answer(&db, sql, &[]);
        if got
            != Ok((
                columns.iter().map(|c| c.to_string()).collect(),
                want.clone(),
            ))
        {
            wrong.push(format!(
                "{sql}\n  got  {got:?}\n  want {columns:?} {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
