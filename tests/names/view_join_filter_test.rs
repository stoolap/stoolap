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

//! A predicate the join pushes to a view on one of its sides filters the
//! view's rows, as it does a table's or a CTE's.

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

#[test]
fn test_a_filter_pushed_to_a_view_side_of_a_join_applies() {
    let db = Database::open("memory://view_join_filter").unwrap();
    db.execute(
        "CREATE TABLE y (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO y VALUES (1, 1, 5), (2, 1, 50), (3, 1, 7), (4, 2, 9), (5, 2, 20)",
        (),
    )
    .unwrap();
    db.execute("CREATE VIEW vy AS SELECT a, b, id FROM y", ())
        .unwrap();
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM vy v JOIN y ON y.id = v.id WHERE v.a = 2"
        ),
        2
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM y JOIN vy v ON y.id = v.id WHERE v.a = 2"
        ),
        2
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM vy v JOIN y ON y.id = v.id WHERE v.a = 2 AND y.b = 20"
        ),
        1
    );
    // and through a correlated EXISTS, where the parent's columns are bound first
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 1, 10), (2, 2, 20), (3, 1, 50)",
        (),
    )
    .unwrap();
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM x WHERE EXISTS (SELECT 1 FROM vy v JOIN y ON y.id = v.id WHERE v.a = x.a AND y.b = x.b)"
        ),
        2
    );
}
