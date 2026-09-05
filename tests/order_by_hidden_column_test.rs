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

//! Sorting by a column the SELECT leaves out reads that column, and the
//! rows go out with the columns the SELECT asked for and no others.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, n INTEGER)",
        (),
    )
    .unwrap();
    for (id, g, n) in [(1i64, "b", 30i64), (2, "a", 10), (3, "b", 20), (4, "a", 40)] {
        db.execute("INSERT INTO t VALUES ($1, $2, $3)", (id, g, n))
            .unwrap();
    }
    db
}

/// The column count the result declares and the width of its rows agree
fn widths(db: &Database, sql: &str) -> (usize, Vec<usize>) {
    let rows = db.query(sql, ()).unwrap();
    let columns = rows.columns().len();
    (columns, rows.map(|row| row.unwrap().len()).collect())
}

fn values(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

#[test]
fn test_sort_key_stays_out_of_the_rows() {
    let db = setup("order_by_hidden_column");
    for sql in [
        "SELECT x.g FROM t x ORDER BY x.id",
        "SELECT x.g FROM t x ORDER BY x.n",
        "SELECT x.g FROM t x ORDER BY x.id DESC",
        "SELECT x.g FROM t x ORDER BY x.n, x.id",
        "SELECT x.g, x.n FROM t x ORDER BY x.id",
        "SELECT g FROM t ORDER BY id",
        "SELECT g FROM t ORDER BY n",
        "SELECT x.g FROM t x ORDER BY x.n LIMIT 2",
        "SELECT x.g FROM t x ORDER BY x.g",
        "SELECT DISTINCT x.g FROM t x ORDER BY x.g",
    ] {
        let (columns, row_widths) = widths(&db, sql);
        assert!(
            row_widths.iter().all(|width| *width == columns),
            "{sql}: {columns} columns, rows {row_widths:?}"
        );
    }
}

#[test]
fn test_sorting_by_a_column_the_select_leaves_out() {
    let db = setup("order_by_hidden_column_values");
    assert_eq!(
        values(&db, "SELECT x.g FROM t x ORDER BY x.n"),
        ["a", "b", "b", "a"],
        "by the column's own order"
    );
    assert_eq!(
        values(&db, "SELECT x.g FROM t x ORDER BY x.n DESC"),
        ["a", "b", "b", "a"],
        "and its reverse"
    );
    assert_eq!(
        values(&db, "SELECT x.g FROM t x ORDER BY x.id"),
        ["b", "a", "b", "a"]
    );
}

/// Sorting by an expression the SELECT leaves out sorts by it
#[test]
fn test_sorting_by_an_expression() {
    let db = Database::open("memory://order_by_expression").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER, f FLOAT, s TEXT)",
        (),
    )
    .unwrap();
    for (id, n, f, s) in [
        (1i64, 10i64, 1.25f64, "b"),
        (2, 20, -2.5, "a"),
        (3, 30, 0.0, "c"),
        (4, 40, 10.0, "d"),
        (5, 50, -0.5, "e"),
    ] {
        db.execute("INSERT INTO t VALUES ($1, $2, $3, $4)", (id, n, f, s))
            .unwrap();
    }
    assert_eq!(
        values(&db, "SELECT id FROM t ORDER BY ABS(f)"),
        ["3", "5", "1", "2", "4"],
        "a function of a column the SELECT leaves out"
    );
    assert_eq!(
        values(&db, "SELECT id FROM t ORDER BY n * -1"),
        ["5", "4", "3", "2", "1"],
        "arithmetic"
    );
    assert_eq!(
        values(&db, "SELECT id FROM t ORDER BY -n"),
        ["5", "4", "3", "2", "1"],
        "a negation"
    );
    assert_eq!(
        values(&db, "SELECT id FROM t ORDER BY s || ''"),
        ["2", "1", "3", "4", "5"],
        "a text expression"
    );
    assert_eq!(
        values(&db, "SELECT id FROM t ORDER BY ABS(f) LIMIT 3"),
        ["3", "5", "1"],
        "with a limit"
    );
    let (columns, row_widths) = widths(&db, "SELECT id FROM t ORDER BY ABS(f)");
    assert!(
        row_widths.iter().all(|width| *width == columns),
        "the expression stays out of the rows: {columns} columns, rows {row_widths:?}"
    );
}

/// A subquery in the ORDER BY reads the row it sorts, not one value for all
#[test]
fn test_sorting_by_a_correlated_subquery() {
    let db = Database::open("memory://order_by_subquery").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, a_id INTEGER)", ())
        .unwrap();
    for id in 1..=5i64 {
        db.execute("INSERT INTO a VALUES ($1)", (id,)).unwrap();
    }
    for (id, a_id) in [(1i64, 1i64), (2, 1), (3, 2), (4, 4)] {
        db.execute("INSERT INTO b VALUES ($1, $2)", (id, a_id))
            .unwrap();
    }
    assert_eq!(
        values(
            &db,
            "SELECT id FROM a ORDER BY (SELECT COUNT(*) FROM b WHERE b.a_id = a.id) DESC, id"
        ),
        ["1", "2", "4", "3", "5"]
    );
    assert_eq!(
        values(
            &db,
            "SELECT id, (SELECT COUNT(*) FROM b WHERE b.a_id = a.id) AS c FROM a ORDER BY c DESC, id"
        )
        .len(),
        5
    );
}

/// An expression naming an alias reads it, since the projection brings the
/// name into scope before the sort runs
#[test]
fn test_sorting_by_an_expression_over_a_select_alias() {
    let db = setup("order_by_alias_expression");
    assert_eq!(
        values(&db, "SELECT n AS x FROM t ORDER BY x + 0"),
        ["10", "20", "30", "40"]
    );
    assert_eq!(
        values(&db, "SELECT n AS x FROM t ORDER BY x * 1 DESC"),
        ["40", "30", "20", "10"]
    );
    assert_eq!(
        values(&db, "SELECT n AS x FROM t ORDER BY ABS(x)"),
        ["10", "20", "30", "40"]
    );
}

/// Inside an expression a source column wins over an alias of the same
/// name; a bare name in ORDER BY reads the alias
#[test]
fn test_a_source_column_wins_over_an_alias_of_the_same_name() {
    let db = setup("order_by_alias_shadow");
    assert_eq!(
        values(&db, "SELECT n AS id FROM t ORDER BY id + 0"),
        ["30", "10", "20", "40"],
        "id + 0 reads the table's id"
    );
    assert_eq!(
        values(&db, "SELECT n AS id FROM t ORDER BY id"),
        ["10", "20", "30", "40"],
        "a bare id reads the alias"
    );
}

/// A function of anything but a bare column is read from the column the
/// projection names after the whole call
#[test]
fn test_sorting_by_a_function_of_an_expression() {
    let db = setup("order_by_function_expression");
    assert_eq!(
        values(&db, "SELECT id FROM t ORDER BY ABS(n - 20)"),
        ["3", "1", "2", "4"]
    );
    assert_eq!(
        values(&db, "SELECT g FROM t ORDER BY ABS(n - 20)"),
        ["b", "b", "a", "a"]
    );
    assert_eq!(
        values(&db, "SELECT id FROM t ORDER BY ABS(n)"),
        ["2", "3", "1", "4"]
    );
    let (columns, widths) = widths(&db, "SELECT id FROM t ORDER BY ABS(n - 20)");
    assert_eq!(columns, 1);
    assert!(widths.iter().all(|w| *w == 1), "{widths:?}");
}
