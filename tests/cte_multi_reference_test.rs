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

//! A CTE stays readable from every place that names it: both branches of a
//! set operation, the select list, HAVING, ORDER BY, a correlated subquery
//! and a later CTE.

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Vec<Vec<Option<i64>>> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap())
        .map(|r| {
            (0..r.len())
                .map(|i| r.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect()
}

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO x VALUES (1, 1), (2, 2), (3, NULL)", ())
        .unwrap();
    db
}

#[test]
fn test_cte_read_by_both_set_operation_branches() {
    let db = setup("cte_multi_reference_set_ops");
    let union = rows(
        &db,
        "WITH c AS (SELECT a FROM x) SELECT a FROM c UNION ALL SELECT a + 1 FROM c",
    );
    assert_eq!(union.len(), 6);
    let counted = rows(
        &db,
        "WITH c AS (SELECT a FROM x) SELECT COUNT(*) \
         FROM (SELECT a FROM c UNION ALL SELECT a + 1 FROM c) u WHERE a IS NOT NULL",
    );
    assert_eq!(counted, [[Some(4)]]);
    let except = rows(
        &db,
        "WITH c AS (SELECT a FROM x) SELECT COUNT(*) \
         FROM (SELECT a FROM c EXCEPT SELECT a + 1 FROM c) u",
    );
    assert_eq!(except, [[Some(1)]]);
}

#[test]
fn test_cte_read_by_select_list_having_and_order_by() {
    let db = setup("cte_multi_reference_clauses");
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x) SELECT (SELECT MAX(a) FROM c) FROM c"
        ),
        [[Some(2)], [Some(2)], [Some(2)]]
    );
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x) SELECT a, COUNT(*) FROM c GROUP BY a \
             HAVING COUNT(*) >= (SELECT MIN(a) FROM c) ORDER BY a"
        )
        .len(),
        3
    );
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x) SELECT a FROM c ORDER BY (SELECT MAX(a) FROM c) - a"
        ),
        [[Some(2)], [Some(1)], [None]]
    );
}

#[test]
fn test_correlated_subquery_over_cte() {
    let db = setup("cte_multi_reference_correlated");
    let counts = [[Some(1), Some(1)], [Some(2), Some(0)], [None, Some(0)]];
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x) \
             SELECT a, (SELECT COUNT(*) FROM c c2 WHERE c2.a > c.a) FROM c ORDER BY a"
        ),
        counts
    );
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x) \
             SELECT c1.a, (SELECT COUNT(*) FROM c c2 WHERE c2.a > c1.a) FROM c c1 ORDER BY a"
        ),
        counts
    );
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x) \
             SELECT a FROM c WHERE EXISTS (SELECT 1 FROM c c2 WHERE c2.a > c.a)"
        ),
        [[Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x) \
             SELECT a FROM c WHERE a IN (SELECT c2.a - 1 FROM c c2 WHERE c2.a > c.a)"
        ),
        [[Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x), \
             d AS (SELECT a FROM c WHERE EXISTS (SELECT 1 FROM c c2 WHERE c2.a > c.a)) \
             SELECT a FROM d"
        ),
        [[Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "WITH c AS (SELECT a FROM x), \
             d AS (SELECT a FROM c ORDER BY (SELECT MAX(a) FROM c) - a LIMIT 1) \
             SELECT a FROM d"
        ),
        [[Some(2)]]
    );
}
