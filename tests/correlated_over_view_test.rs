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

//! A correlated subquery in the select list or the WHERE of a query over a
//! view or a derived table reads each row of that source as its outer row.

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
    db.execute("CREATE VIEW vx AS SELECT id, a FROM x", ())
        .unwrap();
    db
}

#[test]
fn test_correlated_subquery_over_view() {
    let db = setup("correlated_over_view");
    let counts = [[Some(1), Some(1)], [Some(2), Some(0)], [None, Some(0)]];
    assert_eq!(
        rows(
            &db,
            "SELECT a, (SELECT COUNT(*) FROM vx v2 WHERE v2.a > vx.a) FROM vx ORDER BY 1"
        ),
        counts
    );
    assert_eq!(
        rows(
            &db,
            "SELECT v.a, (SELECT COUNT(*) FROM x x2 WHERE x2.a > v.a) FROM vx v ORDER BY 1"
        ),
        counts
    );
    assert_eq!(
        rows(
            &db,
            "SELECT a FROM vx WHERE EXISTS (SELECT 1 FROM vx v2 WHERE v2.a > vx.a)"
        ),
        [[Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT a FROM vx v WHERE (SELECT COUNT(*) FROM x x2 WHERE x2.a > v.a) = 1"
        ),
        [[Some(1)]]
    );
    // A sort column the select list leaves out rides along
    assert_eq!(
        rows(
            &db,
            "SELECT (SELECT COUNT(*) FROM x x2 WHERE x2.a > v.a) FROM vx v ORDER BY v.id DESC"
        ),
        [[Some(0)], [Some(0)], [Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT (SELECT COUNT(*) FROM x x2 WHERE x2.a > d.a) AS n \
             FROM (SELECT id, a FROM x) d ORDER BY d.id DESC, n"
        ),
        [[Some(0)], [Some(0)], [Some(1)]]
    );
}

#[test]
fn test_correlated_where_over_derived_table() {
    let db = setup("correlated_over_derived");
    assert_eq!(
        rows(
            &db,
            "SELECT a FROM (SELECT a FROM x) d WHERE EXISTS (SELECT 1 FROM x x2 WHERE x2.a > d.a)"
        ),
        [[Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT a FROM (SELECT a FROM x) d \
             WHERE a IN (SELECT x2.a - 1 FROM x x2 WHERE x2.a > d.a)"
        ),
        [[Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT a FROM (SELECT a FROM x) d \
             WHERE a > 0 AND (SELECT COUNT(*) FROM x x2 WHERE x2.a > d.a) = 1"
        ),
        [[Some(1)]]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*) FROM x WHERE a IN (SELECT a FROM (SELECT a FROM x) d \
             WHERE EXISTS (SELECT 1 FROM x x2 WHERE x2.a > d.a AND x2.a > x.a))"
        ),
        [[Some(1)]]
    );
}
