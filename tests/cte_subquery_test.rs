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

/// Test CTE references inside derived tables (subqueries)
use stoolap::api::Database;

fn query_count(db: &Database, sql: &str) -> usize {
    let mut count = 0;
    for row in db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("query failed: {}: {}", sql, e))
    {
        let _ = row.expect("row error");
        count += 1;
    }
    count
}

fn setup() -> Database {
    let db = Database::open_in_memory().unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, title TEXT, cat TEXT, val FLOAT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 'A', 'X', 1.0)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, 'B', 'X', 2.0)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, 'C', 'Y', 3.0)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (4, 'D', 'Y', 4.0)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (5, 'E', 'Z', 5.0)", ())
        .unwrap();
    db
}

#[test]
fn test_cte_cross_join_top_level() {
    let db = setup();
    let n = query_count(
        &db,
        "WITH q AS (SELECT 42 AS x) SELECT id, title, q.x FROM t, q",
    );
    assert_eq!(n, 5, "CTE cross join at top level should return 5 rows");
}

#[test]
fn test_cte_cross_join_in_derived_table() {
    let db = setup();
    let n = query_count(
        &db,
        "WITH q AS (SELECT 42 AS x) SELECT * FROM (SELECT id, title, q.x FROM t, q) sub",
    );
    assert_eq!(
        n, 5,
        "CTE cross join inside derived table should return 5 rows"
    );
}

#[test]
fn test_cte_derived_table_with_window() {
    let db = setup();
    let n = query_count(
        &db,
        "WITH q AS (SELECT 42 AS x) \
         SELECT title, cat, rnk FROM ( \
             SELECT title, cat, \
                    RANK() OVER (PARTITION BY cat ORDER BY val) AS rnk \
             FROM t, q \
         ) sub",
    );
    assert_eq!(
        n, 5,
        "CTE + derived table + window function should return 5 rows"
    );
}

#[test]
fn test_derived_table_with_window_no_cte() {
    let db = setup();
    let n = query_count(
        &db,
        "SELECT title, cat, rnk FROM ( \
             SELECT title, cat, \
                    RANK() OVER (PARTITION BY cat ORDER BY val) AS rnk \
             FROM t \
         ) sub",
    );
    assert_eq!(n, 5, "Derived table + window (no CTE) should return 5 rows",);
}

#[test]
fn test_cte_window_with_filter() {
    let db = setup();
    let n = query_count(
        &db,
        "WITH q AS (SELECT 42 AS x) \
         SELECT title, cat FROM ( \
             SELECT title, cat, \
                    RANK() OVER (PARTITION BY cat ORDER BY val) AS rnk \
             FROM t, q \
         ) sub WHERE rnk = 1",
    );
    assert_eq!(
        n, 3,
        "CTE + window + WHERE rnk=1 should return 3 rows (one per category)"
    );
}

#[test]
fn test_window_with_filter_no_cte() {
    let db = setup();
    let n = query_count(
        &db,
        "SELECT title, cat FROM ( \
             SELECT title, cat, \
                    RANK() OVER (PARTITION BY cat ORDER BY val) AS rnk \
             FROM t \
         ) sub WHERE rnk = 1",
    );
    assert_eq!(n, 3, "Window + WHERE rnk=1 (no CTE) should return 3 rows");
}
