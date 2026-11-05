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

//! A statement over CTEs gives the rows the same statement gives over
//! tables holding the CTE results, for every CTE body, reader count, join
//! side and type, outer clause, LIMIT and OFFSET

use stoolap::{Database, Value};

/// The CTE c: its WITH entry, and the query that fills table c
const BODIES: [(&str, &str); 7] = [
    ("c AS (SELECT id, k, v FROM t)", "SELECT id, k, v FROM t"),
    (
        "c AS (SELECT id, k, v FROM t GROUP BY id, k, v)",
        "SELECT id, k, v FROM t",
    ),
    (
        "c AS (SELECT id, k, v FROM t GROUP BY id, k, v ORDER BY id DESC)",
        "SELECT id, k, v FROM t",
    ),
    (
        "c AS (SELECT id, k, v FROM t ORDER BY id LIMIT 5)",
        "SELECT id, k, v FROM t WHERE id <= 5",
    ),
    (
        "c AS (SELECT id, k, v FROM t WHERE id <= 6 UNION ALL SELECT id, k, v FROM t WHERE id > 6)",
        "SELECT id, k, v FROM t",
    ),
    (
        "c(id, k, v) AS (SELECT id, k, v FROM t GROUP BY id, k, v)",
        "SELECT id, k, v FROM t",
    ),
    (
        "c AS (SELECT id, k, v FROM t WHERE id <= 6 GROUP BY id, k, v \
         UNION ALL SELECT id, k, v FROM t WHERE id > 6)",
        "SELECT id, k, v FROM t",
    ),
];

/// Extra WITH entries and the FROM clause, `{J}` standing for the join type
const SIDES: [(&str, &str); 5] = [
    ("", "c a {J} u b ON a.k = b.k"),
    (
        "",
        "c a {J} u b ON a.k = b.k AND (SELECT COUNT(*) FROM c) = (SELECT COUNT(*) FROM t)",
    ),
    ("", "u a {J} c b ON a.k = b.k"),
    ("", "c a {J} c b ON a.k = b.k"),
    (
        ", d AS (SELECT id, k, v FROM c GROUP BY id, k, v)",
        "d a {J} u b ON a.k = b.k",
    ),
];

const JOINS: [&str; 3] = ["JOIN", "LEFT JOIN", "RIGHT JOIN"];

/// Select list, the clauses after FROM, and whether that order is total
const SHAPES: [(&str, &str, bool); 19] = [
    ("a.id, b.id", "", false),
    ("DISTINCT ON (a.k) a.k", "", false),
    ("a.id, b.id", " UNION ALL SELECT id, k FROM u", false),
    ("a.k", " GROUP BY a.k", false),
    ("a.k", " GROUP BY a.k HAVING a.k > 1", false),
    (
        "a.id, 12 IN ((SELECT COUNT(*) FROM c), 99) AS full",
        "",
        false,
    ),
    ("a.id, b.id", " WHERE a.v > 1", false),
    ("a.id, b.id", " ORDER BY a.id, b.id", true),
    ("a.id, b.id", " ORDER BY b.id DESC, a.id DESC", true),
    ("a.id", " ORDER BY a.v, a.id, b.id", true),
    ("DISTINCT a.k", "", false),
    ("DISTINCT ON (a.k) a.k, b.id", " ORDER BY a.k, b.id", true),
    ("a.k, COUNT(*)", " GROUP BY a.k", false),
    ("a.k, COUNT(*)", " GROUP BY ROLLUP(a.k)", false),
    ("a.k, COUNT(*)", " GROUP BY a.k HAVING COUNT(*) > 1", false),
    (
        "a.id, b.id, ROW_NUMBER() OVER (ORDER BY a.id, b.id) AS rn",
        "",
        false,
    ),
    ("COUNT(*)", "", false),
    ("a.id, b.id, (SELECT COUNT(*) FROM c) AS n", "", false),
    (
        "a.id, b.id, (SELECT COUNT(*) FROM c) AS n",
        " ORDER BY a.id, b.id",
        true,
    ),
];

/// The cut, its parameter, and the LIMIT and OFFSET it means
const CUTS: [(&str, Option<i64>, Option<usize>, usize); 8] = [
    (
        " LIMIT 9223372036854775807 OFFSET 1",
        None,
        Some(usize::MAX),
        1,
    ),
    ("", None, None, 0),
    (" LIMIT 0", None, Some(0), 0),
    (" LIMIT 1", None, Some(1), 0),
    (" LIMIT 3", None, Some(3), 0),
    (" LIMIT 3 OFFSET 2", None, Some(3), 2),
    (" LIMIT 2 OFFSET $1", Some(1), Some(2), 1),
    (" LIMIT $1", Some(2), Some(2), 0),
];

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t SELECT g.value, g.value % 4, g.value % 5 FROM generate_series(1, 12) g",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO u SELECT g.value, g.value + 1, g.value % 3 FROM generate_series(1, 6) g",
        (),
    )
    .unwrap();
    db
}

fn rows(db: &Database, sql: &str, param: Option<i64>) -> Result<Vec<String>, String> {
    let result = match param {
        Some(p) => db.query(sql, (p,)),
        None => db.query(sql, ()),
    }
    .map_err(|e| e.to_string())?;
    result
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| match row.get_value(i) {
                    Some(Value::Integer(n)) => n.to_string(),
                    Some(Value::Null(_)) => "NULL".to_string(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

/// Whether `got` is a cut of `reference` the statement's order allows
fn agrees(
    reference: &[String],
    got: &[String],
    ordered: bool,
    cut: (Option<usize>, usize),
) -> bool {
    let (limit, offset) = cut;
    let wanted = reference
        .len()
        .saturating_sub(offset)
        .min(limit.unwrap_or(usize::MAX));
    if ordered {
        return got.iter().eq(reference.iter().skip(offset).take(wanted));
    }
    let mut pool = reference.to_vec();
    got.len() == wanted
        && got
            .iter()
            .all(|row| match pool.iter().position(|r| r == row) {
                Some(at) => {
                    pool.swap_remove(at);
                    true
                }
                None => false,
            })
}

#[test]
fn test_cte_statements_give_the_rows_of_tables() {
    let db = open("cte_clause_matrix");
    let mut found = Vec::new();
    let mut checked = 0;
    for (b, (body, fill)) in BODIES.iter().enumerate() {
        let tables = open(&format!("cte_clause_matrix_tables_{b}"));
        tables
            .execute(&format!("CREATE TABLE c AS {fill}"), ())
            .unwrap();
        tables
            .execute("CREATE TABLE d AS SELECT id, k, v FROM c", ())
            .unwrap();
        for (extra, from) in SIDES {
            for join in JOINS {
                let from = from.replace("{J}", join);
                for (select, tail, ordered) in SHAPES {
                    let reference_sql = format!("SELECT {select} FROM {from}{tail}");
                    let reference = rows(&tables, &reference_sql, None)
                        .unwrap_or_else(|e| panic!("{reference_sql}: {e}"));
                    for (cut, param, limit, offset) in CUTS {
                        let sql =
                            format!("WITH {body}{extra} SELECT {select} FROM {from}{tail}{cut}");
                        checked += 1;
                        match rows(&db, &sql, param) {
                            Err(error) => found.push(format!("{sql}\n    error: {error}")),
                            Ok(got) if !agrees(&reference, &got, ordered, (limit, offset)) => {
                                found.push(format!(
                                    "{sql} [$1 = {param:?}]\n    tables {reference:?}\n    got    {got:?}"
                                ))
                            }
                            Ok(_) => {}
                        }
                    }
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "{} of {checked} statements are not the rows of the tables:\n{}",
        found.len(),
        found.join("\n")
    );
}

/// Select list and the clauses after FROM: each reads c a second time
/// from one position of the statement
const SECOND_READS: [(&str, &str); 7] = [
    ("DISTINCT ON (a.k + (SELECT COUNT(*) FROM c)) a.k", ""),
    ("a.k, COUNT(*)", " GROUP BY a.k, (SELECT COUNT(*) FROM c)"),
    (
        "a.k, COUNT(*)",
        " GROUP BY GROUPING SETS ((a.k, (SELECT COUNT(*) FROM c)))",
    ),
    (
        "a.id, (SELECT STRING_AGG(CAST(t.id AS TEXT), ',' ORDER BY t.id % (SELECT COUNT(*) FROM c), t.id) FROM t) AS s",
        "",
    ),
    (
        "a.id, 'a%' LIKE 'a!%' ESCAPE (SELECT CASE WHEN COUNT(*) = 12 THEN '!' ELSE '#' END FROM c) AS m",
        "",
    ),
    ("a.id, SUM((SELECT COUNT(*) FROM c)) OVER () AS w", ""),
    (
        "a.id, v.n",
        " LEFT JOIN (VALUES ((SELECT COUNT(*) FROM c))) v(n) ON v.n = 12",
    ),
];

#[test]
fn test_a_second_read_of_a_cte_sees_all_of_it() {
    let db = open("cte_clause_matrix_second_reads");
    let tables = open("cte_clause_matrix_second_reads_tables");
    tables
        .execute("CREATE TABLE c AS SELECT id, k, v FROM t", ())
        .unwrap();
    let mut found = Vec::new();
    for (select, tail) in SECOND_READS {
        let from = "c a LEFT JOIN u b ON a.k = b.k";
        let reference_sql = format!("SELECT {select} FROM {from}{tail}");
        let sql = format!(
            "WITH c AS (SELECT id, k, v FROM t GROUP BY id, k, v) SELECT {select} FROM {from}{tail} LIMIT 3"
        );
        // A position no prepared plan takes yet refuses its subquery on both
        let reference = match rows(&tables, &reference_sql, None) {
            Ok(reference) => reference,
            Err(error) => {
                if let Ok(got) = rows(&db, &sql, None) {
                    found.push(format!(
                        "{sql}\n    reference error: {error}\n    got    {got:?}"
                    ));
                }
                continue;
            }
        };
        match rows(&db, &sql, None) {
            // A block reading the CTE is no prepared plan's yet, so it is refused
            Err(error) if error.contains("outside a prepared plan") => {}
            Err(error) => found.push(format!("{sql}\n    error: {error}")),
            Ok(got) if !agrees(&reference, &got, false, (Some(3), 0)) => found.push(format!(
                "{sql}\n    tables {reference:?}\n    got    {got:?}"
            )),
            Ok(_) => {}
        }
    }
    assert!(found.is_empty(), "second reads:\n{}", found.join("\n"));
}
