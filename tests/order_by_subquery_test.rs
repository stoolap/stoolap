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

//! A subquery in ORDER BY is a sort key like any other: a correlated one
//! reads the row it sorts, whether or not the select list carries the
//! columns it reads, and one that reads nothing of the row is run once.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE ox (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE oy (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO ox VALUES (1, 1), (2, 2), (3, 3), (4, NULL), (5, 2)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO oy VALUES (1, 2), (2, 5), (3, 3), (4, 2), (5, NULL)",
        (),
    )
    .unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn test_a_correlated_subquery_sorts_rows_the_select_list_hides_the_key_of() {
    let db = setup("order_by_correlated_subquery");
    let by_matches = "(SELECT COUNT(*) FROM oy WHERE oy.a = ox.a)";
    assert_eq!(
        ids(
            &db,
            &format!("SELECT id FROM ox ORDER BY {by_matches} DESC, id")
        ),
        [2, 5, 3, 1, 4]
    );
    assert_eq!(
        ids(
            &db,
            &format!("SELECT id FROM ox ORDER BY {by_matches} DESC, id LIMIT 3")
        ),
        [2, 5, 3]
    );
    assert_eq!(
        ids(
            &db,
            &format!("SELECT id FROM ox WHERE a IS NOT NULL ORDER BY {by_matches} * 10 + id DESC")
        ),
        [5, 2, 3, 1]
    );
    assert_eq!(
        ids(
            &db,
            &format!("SELECT id, a FROM ox ORDER BY {by_matches} DESC, id")
        ),
        [2, 5, 3, 1, 4]
    );
}

#[test]
fn test_a_subquery_that_reads_nothing_of_the_row_is_a_constant_key() {
    let db = setup("order_by_plain_subquery");
    assert_eq!(
        ids(&db, "SELECT id FROM ox ORDER BY (SELECT 7) - id"),
        [5, 4, 3, 2, 1]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM ox ORDER BY (SELECT MAX(a) FROM oy) - id, id"
        ),
        [5, 4, 3, 2, 1]
    );
}
