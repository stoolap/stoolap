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

//! A correlated subquery whose FROM is a derived table, a view or a CTE
//! binds the parent's column through the parent's table name; the bare
//! name it would otherwise fall back to is the source's own column.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE y (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 1, 10), (2, 2, 20), (3, 3, 30), (4, 1, 40)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO y VALUES (1, 1, 5), (2, 1, 50), (3, 1, 7), (4, 2, 9)",
        (),
    )
    .unwrap();
    db.execute("CREATE VIEW vy AS SELECT a, b FROM y", ())
        .unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

fn values(db: &Database, sql: &str) -> Vec<Option<i64>> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<Option<i64>>(1).unwrap())
        .collect()
}

#[test]
fn test_a_derived_table_source_binds_the_parent_column() {
    let db = setup("correlated_derived");
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM x WHERE EXISTS (SELECT 1 FROM (SELECT a, ROW_NUMBER() OVER (PARTITION BY a ORDER BY id) AS rn FROM y) w WHERE w.a = x.a AND w.rn = 3) ORDER BY id"
        ),
        [1, 4]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM x WHERE EXISTS (SELECT 1 FROM (SELECT a, b FROM y) w WHERE w.a = x.b) ORDER BY id"
        ),
        Vec::<i64>::new()
    );
    assert_eq!(
        values(
            &db,
            "SELECT id, (SELECT COUNT(*) FROM (SELECT a FROM y) w WHERE w.a = x.a) FROM x ORDER BY id"
        ),
        [Some(3), Some(1), Some(0), Some(3)]
    );
}

#[test]
fn test_a_cte_source_binds_the_parent_column() {
    let db = setup("correlated_cte");
    assert_eq!(
        ids(
            &db,
            "WITH t AS (SELECT a, MAX(b) AS mb FROM y GROUP BY a) SELECT id FROM x WHERE b > (SELECT mb FROM t WHERE t.a = x.a) ORDER BY id"
        ),
        [2]
    );
    assert_eq!(
        values(
            &db,
            "WITH t AS (SELECT a, MAX(b) AS mb FROM y GROUP BY a) SELECT id, (SELECT mb FROM t WHERE t.a = x.a) FROM x ORDER BY id"
        ),
        [Some(50), Some(9), None, Some(50)]
    );
}

#[test]
fn test_a_view_source_binds_the_parent_column() {
    let db = setup("correlated_view");
    assert_eq!(
        values(
            &db,
            "SELECT id, (SELECT COUNT(*) FROM vy WHERE vy.a = x.a) FROM x ORDER BY id"
        ),
        [Some(3), Some(1), Some(0), Some(3)]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM x WHERE EXISTS (SELECT 1 FROM vy v WHERE v.a = x.a AND v.b > 40) ORDER BY id"
        ),
        [1, 4]
    );
}
