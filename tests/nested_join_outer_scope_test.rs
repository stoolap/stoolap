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

//! A correlated subquery joining three tables keeps every joined table in
//! its own scope, so only the parent's column is bound from the outer row.

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

#[test]
fn test_three_table_join_inside_correlated_subquery() {
    let db = Database::open("memory://nested_join_outer_scope").unwrap();
    db.execute("CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO x VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    for t in ["p", "q", "r"] {
        db.execute(
            &format!("CREATE TABLE {t} (id INTEGER PRIMARY KEY, a INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO {t} VALUES (1, 1), (2, 2)"), ())
            .unwrap();
    }
    let join = "FROM p JOIN q ON p.id = q.id JOIN r ON q.id = r.id";
    assert_eq!(
        rows(
            &db,
            &format!("SELECT a, (SELECT COUNT(*) {join} WHERE p.a = x.a) FROM x ORDER BY a")
        ),
        [[Some(1), Some(1)], [Some(2), Some(1)], [Some(3), Some(0)]]
    );
    assert_eq!(
        rows(
            &db,
            &format!("SELECT a, (SELECT COUNT(*) {join} WHERE r.a = x.a) FROM x ORDER BY a")
        ),
        [[Some(1), Some(1)], [Some(2), Some(1)], [Some(3), Some(0)]]
    );
    assert_eq!(
        rows(
            &db,
            &format!("SELECT COUNT(*) FROM x WHERE EXISTS (SELECT 1 {join} WHERE p.a = x.a)")
        ),
        [[Some(2)]]
    );
}
