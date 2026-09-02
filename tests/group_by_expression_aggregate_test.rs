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

//! GROUP BY over plain columns, with aggregates that carry an expression, a
//! FILTER or an ORDER BY.
//!
//! These pick the grouping path by the shape of the aggregate rather than by
//! the shape of the key, so they are the queries that move when that choice
//! changes. The results are pinned here.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://group_by_expr_agg_{}", name)).unwrap();
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, dept INTEGER, amount INTEGER)",
        (),
    )
    .unwrap();
    let rows = [
        (1, "east", 1, 100),
        (2, "east", 1, 200),
        (3, "west", 2, 300),
        (4, "west", 2, 400),
        (5, "east", 2, 500),
    ];
    let insert = db
        .prepare("INSERT INTO sales (id, region, dept, amount) VALUES ($1, $2, $3, $4)")
        .unwrap();
    for (id, region, dept, amount) in rows {
        insert
            .execute((id as i64, region, dept as i64, amount as i64))
            .unwrap();
    }
    db
}

/// Collect (key, value) pairs sorted by key, so the assertions do not depend
/// on the order groups come back in.
fn grouped_i64(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    let mut out: Vec<(i64, i64)> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect();
    out.sort();
    out
}

#[test]
fn test_group_by_column_with_expression_aggregate() {
    let db = setup("expr");
    assert_eq!(
        grouped_i64(&db, "SELECT dept, SUM(amount * 2) FROM sales GROUP BY dept"),
        vec![(1, 600), (2, 2400)]
    );
    assert_eq!(
        grouped_i64(&db, "SELECT dept, SUM(amount + 1) FROM sales GROUP BY dept"),
        vec![(1, 302), (2, 1203)]
    );
}

#[test]
fn test_group_by_column_with_filtered_aggregate() {
    let db = setup("filter");
    assert_eq!(
        grouped_i64(
            &db,
            "SELECT dept, COUNT(*) FILTER (WHERE region = 'east') FROM sales GROUP BY dept"
        ),
        vec![(1, 2), (2, 1)]
    );
}

#[test]
fn test_group_by_column_with_expression_aggregate_and_having() {
    let db = setup("having");
    assert_eq!(
        grouped_i64(
            &db,
            "SELECT dept, SUM(amount * 2) FROM sales GROUP BY dept HAVING SUM(amount * 2) > 1000"
        ),
        vec![(2, 2400)]
    );
}

#[test]
fn test_group_by_text_column_with_expression_aggregate() {
    let db = setup("text");
    let mut rows: Vec<(String, i64)> = db
        .query(
            "SELECT region, SUM(amount * 2) FROM sales GROUP BY region",
            (),
        )
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<String>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![("east".to_string(), 1600), ("west".to_string(), 1400)]
    );
}

#[test]
fn test_group_by_column_with_null_group_and_expression_aggregate() {
    let db = Database::open("memory://group_by_expr_agg_nulls").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    let insert = db
        .prepare("INSERT INTO t (id, k, v) VALUES ($1, $2, $3)")
        .unwrap();
    insert.execute((1i64, Option::<i64>::None, 10i64)).unwrap();
    insert.execute((2i64, Option::<i64>::None, 20i64)).unwrap();
    insert.execute((3i64, Some(7i64), 30i64)).unwrap();

    let mut rows: Vec<(Option<i64>, i64)> = db
        .query("SELECT k, SUM(v * 2) FROM t GROUP BY k", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<Option<i64>>(0).unwrap(),
                row.get::<i64>(1).unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(None, 60), (Some(7), 60)]);
}

/// The grouping paths do not agree on whether `0.0` and `-0.0` are one group.
///
/// The storage fast path keys a float column by its bit pattern, so it keeps
/// them apart, and so does the executor path that a query with an expression,
/// FILTER or ORDER BY aggregate takes today. The executor's raw-entry path
/// compares keys with `Value` equality instead, which merges them.
///
/// This pins what each shape answers now. Unifying the paths is worth doing,
/// but it changes group membership rather than just row order, so it has to be
/// a deliberate change with these numbers updated in the same commit, not a
/// side effect of moving a query from one path to another.
#[test]
fn test_signed_zero_group_membership_is_stable_per_shape() {
    let db = Database::open("memory://group_by_expr_agg_signed_zero").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v FLOAT, w INTEGER)",
        (),
    )
    .unwrap();
    let insert = db
        .prepare("INSERT INTO t (id, v, w) VALUES ($1, $2, $3)")
        .unwrap();
    insert.execute((1i64, 0.0f64, 10i64)).unwrap();
    insert.execute((2i64, -0.0f64, 20i64)).unwrap();
    insert.execute((3i64, 1.0f64, 30i64)).unwrap();

    let groups = |sql: &str| db.query(sql, ()).unwrap().count();

    assert_eq!(groups("SELECT v, SUM(w) FROM t GROUP BY v"), 3);
    assert_eq!(groups("SELECT v, COUNT(*) FROM t GROUP BY v"), 3);
    assert_eq!(groups("SELECT v, SUM(w * 1) FROM t GROUP BY v"), 3);
}
