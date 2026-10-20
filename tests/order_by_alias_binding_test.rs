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

//! A bare ORDER BY name that is both a SELECT alias and an input column
//! sorts by the output column; a qualified name or an expression sorts by
//! the input column, on every path

use stoolap::Database;

fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        "CREATE INDEX t_k ON t (k)",
        "INSERT INTO t VALUES (1, 1, 1), (2, 2, 1), (3, 3, 2)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY)",
        "INSERT INTO s VALUES (1), (2), (3)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY)",
        "CREATE TABLE w (a INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO w VALUES (1, 30), (2, 20), (3, 10)",
        "CREATE TABLE ja (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO ja VALUES (1, 1), (2, 2), (3, 3)",
        "CREATE TABLE jb (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO jb VALUES (1, 30), (2, 20), (3, 10)",
        "CREATE TABLE sa (id INTEGER PRIMARY KEY, v INTEGER, bid INTEGER)",
        "INSERT INTO sa VALUES (1, 1, 30), (2, 2, 20), (3, 3, 10)",
        "CREATE TABLE sb (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO sb VALUES (10, 0), (20, 0), (30, 0)",
        "CREATE VIEW v AS SELECT id FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn first_column(db: &Database, sql: &str, params: &[i64]) -> Result<Vec<i64>, String> {
    let params: Vec<stoolap::Value> = params.iter().map(|&p| stoolap::Value::Integer(p)).collect();
    db.query(sql, params)
        .and_then(|rows| {
            rows.map(|row| row.and_then(|row| row.get::<i64>(0)))
                .collect::<stoolap::Result<Vec<_>>>()
        })
        .map_err(|e| e.to_string())
}

#[test]
fn an_order_by_name_binds_the_output_column_and_a_qualified_one_the_input() {
    let db = tables("order_by_alias_binding");
    let output = vec![-3, -2, -1];
    let input = vec![-1, -2, -3];
    let mut wrong = Vec::new();
    for (sql, params, want) in [
        (
            "SELECT -id AS id FROM t ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "SELECT -t.id AS id FROM t ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "SELECT -id AS id FROM t ORDER BY id LIMIT 2",
            vec![],
            vec![-3, -2],
        ),
        (
            "SELECT -id AS id FROM t ORDER BY id LIMIT 2 OFFSET 1",
            vec![],
            vec![-2, -1],
        ),
        (
            "SELECT -id AS id FROM t ORDER BY id LIMIT $1",
            vec![2],
            vec![-3, -2],
        ),
        (
            "SELECT -id AS id FROM t ORDER BY id DESC LIMIT 2",
            vec![],
            vec![-1, -2],
        ),
        (
            "SELECT -k AS k FROM t ORDER BY k LIMIT 2",
            vec![],
            vec![-3, -2],
        ),
        (
            "SELECT -k AS k FROM t WHERE k > 0 ORDER BY k",
            vec![],
            output.clone(),
        ),
        (
            "SELECT DISTINCT -id AS id FROM t ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "SELECT -id AS id, ROW_NUMBER() OVER (ORDER BY id) FROM t ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "SELECT -id AS id FROM (SELECT id FROM t) d ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "WITH c AS (SELECT id FROM t) SELECT -c.id AS id FROM c ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "WITH c AS (SELECT id FROM t) SELECT -c.id AS id FROM c ORDER BY id LIMIT 2",
            vec![],
            vec![-3, -2],
        ),
        (
            "SELECT -t.id AS id FROM t JOIN s ON s.id = t.id ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "SELECT -t.id AS id FROM t JOIN s ON s.id = t.id ORDER BY id LIMIT 2",
            vec![],
            vec![-3, -2],
        ),
        (
            "SELECT -t.id AS id FROM t LEFT JOIN b ON b.id = t.id ORDER BY id",
            vec![],
            output.clone(),
        ),
        (
            "SELECT -id AS id FROM t ORDER BY t.id",
            vec![],
            input.clone(),
        ),
        (
            "SELECT -id AS id FROM t ORDER BY t.id LIMIT 2",
            vec![],
            vec![-1, -2],
        ),
        (
            "SELECT -t.id AS id FROM t JOIN s ON s.id = t.id ORDER BY t.id",
            vec![],
            input.clone(),
        ),
        (
            "SELECT -id AS id FROM t ORDER BY id + 0",
            vec![],
            input.clone(),
        ),
        ("SELECT -id AS n FROM t ORDER BY id", vec![], input.clone()),
        ("SELECT -id AS n FROM t ORDER BY n", vec![], output.clone()),
        ("WITH c AS (SELECT id FROM t) SELECT -c.id AS id FROM c ORDER BY c.id", vec![], input.clone()),
        (
            "WITH c AS (SELECT id, k FROM t) SELECT -id AS id, k FROM c ORDER BY c.id",
            vec![],
            input.clone(),
        ),
        (
            "SELECT v.*, -id AS x, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM v ORDER BY x",
            vec![],
            vec![3, 2, 1],
        ),
        (
            "SELECT t.*, -t.id AS x, ROW_NUMBER() OVER (ORDER BY t.id) AS rn FROM t JOIN s ON s.id = t.id ORDER BY x",
            vec![],
            vec![3, 2, 1],
        ),
        (
            "WITH c AS (SELECT id FROM t) SELECT c.*, -id AS x, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM c ORDER BY x",
            vec![],
            vec![3, 2, 1],
        ),
        (
            "SELECT -id AS id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t ORDER BY t.id",
            vec![],
            input.clone(),
        ),
        (
            "WITH c(id) AS (SELECT id FROM t) SELECT -id AS x FROM c ORDER BY (SELECT 0) + id",
            vec![],
            input.clone(),
        ),
        (
            "WITH c(id) AS (SELECT id FROM t) SELECT id AS x FROM c ORDER BY id, (SELECT -c.id)",
            vec![],
            vec![1, 2, 3],
        ),
        (
            "WITH c(id) AS (SELECT id FROM t) SELECT id AS x, id * 2 AS id FROM c ORDER BY (SELECT -3 * c.id) + id * 2",
            vec![],
            vec![3, 2, 1],
        ),
        (
            "SELECT \"ǅ\".*, -id AS rev FROM t AS \"ǅ\" ORDER BY rev",
            vec![],
            vec![3, 2, 1],
        ),
        (
            "WITH g (id, n) AS (SELECT id, COUNT(*) FROM t GROUP BY id) SELECT -g.id AS id FROM g LEFT JOIN s ON s.id = g.id ORDER BY id LIMIT 2",
            vec![],
            vec![-3, -2],
        ),
        (
            "WITH c AS (SELECT id FROM t) SELECT -c.id AS id FROM c JOIN s ON s.id = c.id ORDER BY id LIMIT 2",
            vec![],
            vec![-3, -2],
        ),
        ("SELECT b AS a, a AS b FROM w ORDER BY a LIMIT 1", vec![], vec![10]),
        ("SELECT b AS a, a AS b FROM w ORDER BY w.a LIMIT 1", vec![], vec![30]),
        ("SELECT *, -id AS id FROM t ORDER BY t.id", vec![], vec![1, 2, 3]),
        ("SELECT -id AS id, * FROM t ORDER BY t.id", vec![], input.clone()),
        ("SELECT *, -id AS id FROM t ORDER BY id", vec![], vec![3, 2, 1]),
        (
            "SELECT jb.k AS k FROM ja JOIN jb ON jb.id = ja.id ORDER BY k LIMIT 2",
            vec![],
            vec![10, 20],
        ),
        ("SELECT jb.k AS k FROM ja JOIN jb ON jb.id = ja.id ORDER BY k", vec![], vec![10, 20, 30]),
        (
            "SELECT sa.*, sb.id, sb.k AS k FROM sa JOIN sb ON sb.id = sa.bid ORDER BY k, sb.id LIMIT 2",
            vec![],
            vec![3, 2],
        ),
        (
            "SELECT sa.*, sb.id, sb.k AS k FROM sa JOIN sb ON sb.id = sa.bid ORDER BY k, sb.id",
            vec![],
            vec![3, 2, 1],
        ),
        ("SELECT *, id * 0 AS z FROM t ORDER BY z, t.k DESC", vec![], vec![3, 2, 1]),
        ("SELECT t.id AS id FROM t ORDER BY id LIMIT 2", vec![], vec![1, 2]),
        ("SELECT -id AS x, t.id AS x FROM t ORDER BY t.id", vec![], input.clone()),
        ("SELECT -id AS x, id AS x FROM t ORDER BY id", vec![], input.clone()),
        ("SELECT -id AS x, *, t.id AS y FROM t ORDER BY t.id DESC", vec![], output.clone()),
        ("SELECT -id AS x, *, t.id AS x FROM t ORDER BY t.id", vec![], input.clone()),
        ("SELECT -id AS x, *, id AS x FROM t ORDER BY id", vec![], input.clone()),
        ("SELECT *, -id AS id, t.id AS x FROM t ORDER BY t.id DESC", vec![], vec![3, 2, 1]),
        (
            "SELECT *, -id AS id, t.id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t ORDER BY id, t.k",
            vec![],
            vec![3, 2, 1],
        ),
        ("SELECT -id AS x, *, *, t.id AS x FROM t ORDER BY t.id", vec![], input.clone()),
        ("SELECT t.*, -id AS id, * FROM t ORDER BY id", vec![], vec![3, 2, 1]),
    ] {
        let got = first_column(&db, sql, &params);
        if got != Ok(want.clone()) {
            wrong.push(format!("{sql}\n  got  {got:?}\n  want {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_grouped_order_by_name_binds_the_output_column() {
    let db = tables("order_by_alias_grouped");
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT -MAX(id) AS id FROM t GROUP BY g ORDER BY id",
            vec![-3, -2],
        ),
        ("SELECT -g AS g FROM t GROUP BY g ORDER BY g", vec![-2, -1]),
        (
            "SELECT -g AS g FROM t GROUP BY g ORDER BY g LIMIT 1",
            vec![-2],
        ),
    ] {
        let got = first_column(&db, sql, &[]);
        if got != Ok(want.clone()) {
            wrong.push(format!("{sql}\n  got  {got:?}\n  want {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_prepared_order_by_name_binds_the_output_column() {
    let db = tables("order_by_alias_prepared");
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT -id AS id FROM t WHERE id > $1 ORDER BY id",
            vec![-3, -2, -1],
        ),
        (
            "SELECT -id AS id FROM t WHERE id > $1 ORDER BY id LIMIT 2",
            vec![-3, -2],
        ),
        (
            "SELECT -id AS id FROM t WHERE id > $1 ORDER BY t.id LIMIT 2",
            vec![-1, -2],
        ),
    ] {
        let statement = db.prepare(sql).unwrap();
        let got = statement
            .query((0,))
            .and_then(|rows| {
                rows.map(|row| row.and_then(|row| row.get::<i64>(0)))
                    .collect::<stoolap::Result<Vec<_>>>()
            })
            .map_err(|e| e.to_string());
        if got != Ok(want.clone()) {
            wrong.push(format!("{sql}\n  got  {got:?}\n  want {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn an_alias_over_its_own_column_reads_as_the_column_does() {
    use stoolap::test_failpoints::take_read_paths;
    let db = tables("order_by_alias_own_column");
    let mut wrong = Vec::new();
    for (plain, aliased) in [
        (
            "SELECT id FROM t ORDER BY id LIMIT 1",
            "SELECT t.id AS id FROM t ORDER BY id LIMIT 1",
        ),
        (
            "SELECT k FROM t ORDER BY k LIMIT 1",
            "SELECT t.k AS k FROM t ORDER BY k LIMIT 1",
        ),
    ] {
        take_read_paths();
        let want = first_column(&db, plain, &[]);
        let want_paths = take_read_paths();
        let got = first_column(&db, aliased, &[]);
        let got_paths = take_read_paths();
        if got != want || got_paths != want_paths {
            wrong.push(format!(
                "{aliased}\n  got  {got:?} {got_paths:?}\n  want {want:?} {want_paths:?}"
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

#[test]
fn the_rule_holds_for_every_source_and_pipeline() {
    let db = Database::open("memory://order_by_alias_matrix").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE INDEX t_k ON t (k)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, z INTEGER)",
        "INSERT INTO s VALUES (1, 0), (2, 0), (3, 0)",
        "CREATE VIEW v AS SELECT id, k FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // (name, WITH, FROM, WHERE, whether a qualified star expands there)
    let sources = [
        ("table", "", "t AS src", "", true),
        ("join", "", "t AS src JOIN s ON s.id = src.id", "", true),
        ("view", "", "v AS src", "", false),
        ("derived", "", "(SELECT id, k FROM t) AS src", "", false),
        (
            "cte",
            "WITH c AS (SELECT id, k FROM t) ",
            "c AS src",
            "",
            false,
        ),
        (
            "cte_columns",
            "WITH c(id, k) AS (SELECT id, k FROM t) ",
            "c AS src",
            "",
            false,
        ),
        (
            "cte_materialized",
            "WITH c AS (SELECT id, k FROM t) ",
            "c AS src",
            "src.id IN (SELECT id FROM c)",
            false,
        ),
        (
            "values",
            "",
            "(SELECT column1 AS id, column2 AS k FROM (VALUES (1, 10), (2, 20), (3, 30))) AS src",
            "",
            false,
        ),
    ];
    let output = vec![3, 2, 1];
    let input = vec![1, 2, 3];
    // (SELECT items after src.id, ORDER BY, ids in order, whether a star is used)
    let shapes = [
        ("-k AS k", "k", &output, 0),
        ("-k AS k", "src.k", &input, 0),
        ("-k AS k", "k + 0", &input, 0),
        ("-k AS k", "2", &output, 0),
        ("-k AS k", "k DESC, src.id", &input, 0),
        ("k AS x", "k", &input, 0),
        ("k AS x", "k, k * 2", &input, 0),
        ("k AS x", "x DESC", &output, 0),
        ("-k AS x", "x * 1", &output, 0),
        ("-k AS x", "k", &input, 0),
        ("*, -k AS k", "k", &output, 1),
        ("*, -k AS k", "src.k", &input, 1),
        ("*, -k AS k", "k + 0", &input, 1),
        ("src.*, -k AS k", "k", &output, 2),
        ("src.*, -k AS k", "src.k", &input, 2),
        ("-k AS x", "(SELECT 0) + k", &input, 0),
        ("-k AS k", "(SELECT 0) + k", &input, 0),
        ("k AS x", "(SELECT 0) + k", &input, 0),
        ("-k AS x", "(SELECT src.k)", &input, 0),
        ("-k AS k", "(SELECT src.k)", &input, 0),
        ("k AS x", "k, (SELECT -src.k)", &input, 0),
        ("k AS x", "(SELECT -src.k) DESC", &input, 0),
        ("-k AS k", "(SELECT 0 * src.k) + k", &input, 0),
        (
            "k AS x, k * 2 AS k",
            "(SELECT -3 * src.k) + k * 2",
            &output,
            0,
        ),
    ];
    // (name, DISTINCT, extra item, GROUP BY, LIMIT/OFFSET, prepared)
    let pipelines = [
        ("plain", false, false, false, "", false),
        ("limit", false, false, false, " LIMIT 3", false),
        ("offset", false, false, false, " LIMIT 3 OFFSET 0", false),
        ("distinct", true, false, false, "", false),
        ("window", false, true, false, "", false),
        ("window_limit", false, true, false, " LIMIT 3", false),
        ("grouped", false, false, true, "", false),
        ("grouped_window", false, true, true, "", false),
        ("prepared", false, false, false, " LIMIT 3", true),
    ];
    let mut wrong = Vec::new();
    let mut ran = 0;
    for (source, with, from, filter, qualified_star) in sources {
        for (items, order_by, want, star) in shapes {
            if star == 2 && !qualified_star {
                continue;
            }
            for (pipeline, distinct, window, grouped, rows, prepared) in pipelines {
                // Out of this rule: a star under GROUP BY is refused, an alias
                // named as a grouping column is #508, a grouped window over a
                // view or a materialized CTE is #509, and DISTINCT ordered by
                // a column it does not select is not this rule's to settle
                if grouped && (star > 0 || items == "-k AS k")
                    || grouped
                        && window
                        && matches!(source, "view" | "cte_columns" | "cte_materialized")
                    || distinct && items == "-k AS x" && order_by == "k"
                {
                    continue;
                }
                let mut conditions: Vec<&str> = Vec::new();
                if !filter.is_empty() {
                    conditions.push(filter);
                }
                if prepared {
                    conditions.push("src.id > $1");
                }
                let sql =
                    format!(
                    "{with}SELECT {}src.id, {items}{} FROM {from}{}{} ORDER BY {order_by}{rows}",
                    if distinct { "DISTINCT " } else { "" },
                    if window {
                        ", ROW_NUMBER() OVER (ORDER BY src.id) AS rn"
                    } else {
                        ""
                    },
                    if conditions.is_empty() {
                        String::new()
                    } else {
                        format!(" WHERE {}", conditions.join(" AND "))
                    },
                    if grouped { " GROUP BY src.id, src.k" } else { "" },
                );
                let got = if prepared {
                    db.prepare(&sql)
                        .and_then(|statement| {
                            statement.query((0,)).and_then(|rows| {
                                rows.map(|row| row.and_then(|row| row.get::<i64>(0)))
                                    .collect::<stoolap::Result<Vec<_>>>()
                            })
                        })
                        .map_err(|e| e.to_string())
                } else {
                    first_column(&db, &sql, &[])
                };
                ran += 1;
                if got.as_ref() != Ok(want) {
                    wrong.push(format!(
                        "[{source}/{pipeline}] {sql}\n  got  {got:?}\n  want {want:?}"
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
