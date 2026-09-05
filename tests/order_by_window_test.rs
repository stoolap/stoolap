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

//! A window function in ORDER BY is a sort key like any other, whether or
//! not the select list carries it, beside a star included.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE hx (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO hx VALUES (1, 1, 10), (2, 1, 11), (3, 2, 50), (4, 3, 30), (5, NULL, 40), (6, 3, 5)",
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
fn test_a_window_in_order_by_sorts_the_rows() {
    let db = setup("order_by_window");
    // by b descending: 3, 5, 4, 2, 1, 6
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM hx ORDER BY ROW_NUMBER() OVER (ORDER BY b DESC)"
        ),
        [3, 5, 4, 2, 1, 6]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM hx WHERE b > 8 ORDER BY RANK() OVER (ORDER BY b DESC), id LIMIT 3"
        ),
        [3, 5, 4]
    );
    // sums by a: 2 gives 50, NULL 40, 3 gives 35, 1 gives 21
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM hx ORDER BY SUM(b) OVER (PARTITION BY a) DESC, id"
        ),
        [3, 5, 4, 6, 1, 2]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id, ROW_NUMBER() OVER (ORDER BY b DESC) FROM hx ORDER BY ROW_NUMBER() OVER (ORDER BY b DESC) DESC"
        ),
        [6, 1, 2, 4, 5, 3]
    );
}

#[test]
fn test_the_hidden_window_column_is_dropped_beside_a_star() {
    let db = setup("order_by_window_star");
    let result = db
        .query(
            "SELECT * FROM hx ORDER BY ROW_NUMBER() OVER (ORDER BY b DESC) LIMIT 2",
            (),
        )
        .unwrap();
    assert_eq!(result.columns(), ["id", "a", "b"]);
    let rows: Vec<Vec<Option<i64>>> = result
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| r.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect();
    assert_eq!(
        rows,
        [
            vec![Some(3), Some(2), Some(50)],
            vec![Some(5), None, Some(40)]
        ]
    );
}
