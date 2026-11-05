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

//! An unaliased window function column is named as the query wrote it,
//! its OVER clause included.

use stoolap::Database;

#[test]
fn test_an_unaliased_window_column_is_named_as_written() {
    let db = Database::open("memory://window_column_names").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, f FLOAT, s TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t (g)", ()).unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 1.5, 'a'), (2, 1, 2.5, 'b'), (3, 2, 3.5, 'c')",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT SUM(f) OVER (), COUNT(*) OVER (PARTITION BY g) FROM t",
            vec!["SUM(f) OVER ()", "COUNT(*) OVER (PARTITION BY g)"],
        ),
        (
            "SELECT ROW_NUMBER() OVER (ORDER BY id), RANK() OVER (ORDER BY f DESC) FROM t",
            vec![
                "ROW_NUMBER() OVER (ORDER BY id)",
                "RANK() OVER (ORDER BY f DESC)",
            ],
        ),
        (
            "SELECT SUM(f) OVER (), SUM(f) OVER (PARTITION BY g) FROM t",
            vec!["SUM(f) OVER ()", "SUM(f) OVER (PARTITION BY g)"],
        ),
        (
            "SELECT LAG(s, 1) OVER (ORDER BY id NULLS FIRST) FROM t",
            vec!["LAG(s, 1) OVER (ORDER BY id NULLS FIRST)"],
        ),
        (
            "SELECT SUM(f) OVER w FROM t WINDOW w AS (PARTITION BY g)",
            vec!["SUM(f) OVER w"],
        ),
        (
            "SELECT SUM(f) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
            vec!["SUM(f) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"],
        ),
        (
            "SELECT g, SUM(f) OVER (PARTITION BY g) FROM t LIMIT 2",
            vec!["g", "SUM(f) OVER (PARTITION BY g)"],
        ),
        (
            "SELECT g, SUM(SUM(f)) OVER () FROM t GROUP BY g",
            vec!["g", "SUM(SUM(f)) OVER ()"],
        ),
        ("SELECT SUM(f) OVER () AS total FROM t", vec!["total"]),
    ] {
        let columns = db
            .query(sql, ())
            .map(|rows| rows.columns().to_vec())
            .map_err(|e| e.to_string());
        if columns.as_deref()
            != Ok(want
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .as_slice())
        {
            wrong.push(format!("{sql} named {columns:?}, want {want:?}"));
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
fn test_a_sort_window_is_not_taken_for_a_visible_one_named_alike() {
    let db = Database::open("memory://window_sort_identity").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10), (2, 1, -100), (3, 2, 1)",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT g, SUM(v) FILTER (WHERE v > 0) OVER (PARTITION BY g) FROM t \
             ORDER BY SUM(v) OVER (PARTITION BY g), id",
            vec![1, 1, 2],
        ),
        (
            "SELECT g, SUM(v) FILTER (WHERE v > 0) OVER (PARTITION BY g) FROM t \
             ORDER BY SUM(v) OVER (PARTITION BY g) DESC, id",
            vec![2, 1, 1],
        ),
    ] {
        let got: Result<(usize, Vec<i64>), String> =
            db.query(sql, ()).map_err(|e| e.to_string()).map(|rows| {
                let width = rows.columns().len();
                let groups = rows
                    .map(|row| row.unwrap().get::<i64>(0).unwrap())
                    .collect();
                (width, groups)
            });
        if got != Ok((2, want.clone())) {
            wrong.push(format!(
                "{sql} gave {got:?}, want 2 columns and groups {want:?}"
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
fn test_distinct_on_reads_a_sort_window_by_identity() {
    let db = Database::open("memory://window_distinct_on_identity").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 5), (2, 5), (3, 6)", ())
        .unwrap();
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT DISTINCT ON (ROW_NUMBER() OVER (ORDER BY id)) id FROM t \
             ORDER BY ROW_NUMBER() OVER (ORDER BY id)",
            vec![1, 2, 3],
        ),
        (
            "SELECT DISTINCT ON (DENSE_RANK() OVER (ORDER BY v)) id FROM t \
             ORDER BY DENSE_RANK() OVER (ORDER BY v), id",
            vec![1, 3],
        ),
    ] {
        let got: Result<Vec<i64>, String> =
            db.query(sql, ()).map_err(|e| e.to_string()).map(|rows| {
                rows.map(|row| row.unwrap().get::<i64>(0).unwrap())
                    .collect()
            });
        if got.as_ref() != Ok(&want) {
            wrong.push(format!("{sql} gave {got:?}, want {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
