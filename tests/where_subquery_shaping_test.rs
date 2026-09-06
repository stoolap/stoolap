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

//! A subquery in the WHERE clause narrows the rows. Everything the query
//! says about the rows it keeps still applies: what to count, what order to
//! return them in, how to group them, and what to number them by.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, g TEXT, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO s VALUES (1,'a',10), (2,'a',20), (3,'b',30), (4,'b',40), (5,'c',50)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, s_id INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1, 1), (2, 2)", ())
        .unwrap();
    db
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
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

const NOT_EXISTS: &str = "NOT EXISTS (SELECT 1 FROM p WHERE p.s_id = s.id)";
const EXISTS: &str = "EXISTS (SELECT 1 FROM p WHERE p.s_id = s.id)";
const NOT_IN: &str = "id NOT IN (SELECT s_id FROM p WHERE s_id IS NOT NULL)";

/// An aggregate over the rows a NOT EXISTS keeps
#[test]
fn test_aggregate_over_a_not_exists() {
    let db = setup("where_sub_shape_aggregate");
    assert_eq!(
        rows(&db, &format!("SELECT COUNT(*) FROM s WHERE {NOT_EXISTS}")),
        ["3"]
    );
    assert_eq!(
        rows(&db, &format!("SELECT SUM(n) FROM s WHERE {NOT_EXISTS}")),
        ["120"]
    );
}

/// Grouping the rows a NOT EXISTS keeps
#[test]
fn test_group_by_over_a_not_exists() {
    let db = setup("where_sub_shape_group");
    assert_eq!(
        rows(
            &db,
            &format!("SELECT g, COUNT(*) FROM s WHERE {NOT_EXISTS} GROUP BY g ORDER BY g")
        ),
        ["b,2", "c,1"]
    );
    assert_eq!(
        rows(
            &db,
            &format!("SELECT g FROM s WHERE {NOT_EXISTS} GROUP BY g HAVING COUNT(*) > 1")
        ),
        ["b"]
    );
}

/// Sorting by a column the SELECT list leaves out
#[test]
fn test_order_by_a_column_the_select_leaves_out() {
    let db = setup("where_sub_shape_order");
    for predicate in [NOT_EXISTS, NOT_IN] {
        assert_eq!(
            rows(
                &db,
                &format!("SELECT id FROM s WHERE {predicate} ORDER BY n DESC")
            ),
            ["5", "4", "3"],
            "{predicate}"
        );
    }
    assert_eq!(
        rows(
            &db,
            &format!("SELECT id FROM s WHERE {EXISTS} ORDER BY n DESC")
        ),
        ["2", "1"]
    );
}

/// Numbering the rows a subquery keeps
#[test]
fn test_window_function_over_a_subquery_predicate() {
    let db = setup("where_sub_shape_window");
    for predicate in [NOT_EXISTS, NOT_IN] {
        assert_eq!(
            rows(
                &db,
                &format!("SELECT ROW_NUMBER() OVER (ORDER BY id) FROM s WHERE {predicate}")
            ),
            ["1", "2", "3"],
            "{predicate}"
        );
    }
    assert_eq!(
        rows(
            &db,
            &format!("SELECT ROW_NUMBER() OVER (ORDER BY id) FROM s WHERE {EXISTS}")
        ),
        ["1", "2"]
    );
}
