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
