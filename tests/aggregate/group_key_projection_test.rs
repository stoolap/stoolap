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

//! A select column that is an expression over a GROUP BY key is evaluated
//! for every group, the NULL group included, while a column that repeats a
//! GROUP BY expression is read from the group as it is.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE hx (a INTEGER, b INTEGER, s TEXT)", ())
        .unwrap();
    db.execute(
        "INSERT INTO hx VALUES (1, 10, 'p'), (1, 11, 'p'), (2, 20, 'q'), (3, 30, NULL), (NULL, 40, 'q')",
        (),
    )
    .unwrap();
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| {
                    r.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    out.sort();
    out
}

#[test]
fn test_a_function_of_the_group_key_is_evaluated() {
    let db = setup("group_key_function");
    assert_eq!(
        rows(&db, "SELECT COALESCE(a, 5) FROM hx GROUP BY a"),
        ["1", "2", "3", "5"]
    );
    assert_eq!(
        rows(&db, "SELECT IFNULL(a, 5), SUM(b) FROM hx GROUP BY a"),
        ["1,21", "2,20", "3,30", "5,40"]
    );
    assert_eq!(
        rows(&db, "SELECT NULLIF(a, 1) FROM hx GROUP BY a"),
        ["2", "3", "NULL", "NULL"]
    );
    assert_eq!(
        rows(&db, "SELECT COALESCE(s, '-'), COUNT(*) FROM hx GROUP BY s"),
        ["-,1", "p,2", "q,2"]
    );
}

#[test]
fn test_a_literal_or_cast_beside_the_group_key_is_evaluated() {
    let db = setup("group_key_literal");
    assert_eq!(
        rows(&db, "SELECT 1, COUNT(*) FROM hx GROUP BY a"),
        ["1,1", "1,1", "1,1", "1,2"]
    );
    assert_eq!(
        rows(&db, "SELECT CAST(a AS TEXT), COUNT(*) FROM hx GROUP BY a"),
        ["1,2", "2,1", "3,1", "NULL,1"]
    );
}

#[test]
fn test_a_group_by_expression_column_is_read_as_it_is() {
    let db = setup("group_key_expression");
    assert_eq!(
        rows(
            &db,
            "SELECT UPPER(s), COUNT(*), SUM(b) FROM hx GROUP BY UPPER(s)"
        ),
        ["NULL,1,30", "P,2,21", "Q,2,60"]
    );
    assert_eq!(
        rows(&db, "SELECT LENGTH(s), COUNT(*) FROM hx GROUP BY LENGTH(s)"),
        ["1,4", "NULL,1"]
    );
}

#[test]
fn test_counting_alone_over_a_text_key_of_a_derived_table() {
    let db = setup("group_key_derived_text");
    assert_eq!(
        rows(&db, "SELECT COUNT(*) FROM (SELECT s FROM hx) t GROUP BY s"),
        ["1", "2", "2"]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*) FROM (SELECT '' AS t UNION ALL SELECT '' UNION ALL SELECT NULL) GROUP BY t"
        ),
        ["1", "2"]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*), s FROM (SELECT s FROM hx) t GROUP BY s"
        ),
        ["1,NULL", "2,p", "2,q"]
    );
}

#[test]
fn test_a_derived_group_by_keeps_every_select_expression() {
    let db = Database::open("memory://group_key_projection_derived_exprs").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'a'), (3, 'b')", ())
        .unwrap();
    // An expression over the group column beside the aggregates is a
    // column of its own, whatever path answers the query
    let rows: Vec<(String, String, i64)> = db
        .query(
            "SELECT k, UPPER(k), COUNT(*) FROM (SELECT k FROM t) s GROUP BY k ORDER BY k",
            (),
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<String>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        [("a".into(), "A".into(), 2), ("b".into(), "B".into(), 1)]
    );
}

#[test]
fn test_a_repeated_aggregate_beside_a_hidden_order_by_aggregate_keeps_its_place() {
    let db = Database::open("memory://group_key_projection_hidden_aggregate").unwrap();
    // COUNT(*) appears twice and SUM(v) only in ORDER BY: the parser folds
    // the repeat and appends the hidden one, so the select list and the
    // parsed aggregates disagree on positions
    let rows: Vec<(String, i64, i64)> = db
        .query(
            "SELECT t.k, COUNT(*) AS a, COUNT(*) AS b FROM (SELECT 'x' AS k, 10 AS v) t \
             GROUP BY t.k ORDER BY SUM(v)",
            (),
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<String>(0).unwrap(),
                r.get::<i64>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();
    assert_eq!(rows, [("x".into(), 1, 1)]);
}
