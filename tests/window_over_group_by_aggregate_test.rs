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

//! A window function over a GROUP BY result may order or partition by an
//! aggregate the select list does not carry; the aggregate is computed
//! for it.

use stoolap::Database;

fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap())
        })
        .collect()
}

#[test]
fn test_a_window_orders_by_an_aggregate_outside_the_select_list() {
    let db = Database::open("memory://window_over_group_by_aggregate").unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 1, 10), (2, 2, 20), (3, 3, 30), (4, 1, 40)",
        (),
    )
    .unwrap();
    // sums: a = 1 gives 50, a = 3 gives 30, a = 2 gives 20
    assert_eq!(
        pairs(
            &db,
            "SELECT a, ROW_NUMBER() OVER (ORDER BY SUM(b) DESC, a) AS r FROM x GROUP BY a ORDER BY r"
        ),
        [(1, 1), (3, 2), (2, 3)]
    );
    assert_eq!(
        pairs(
            &db,
            "SELECT a, RANK() OVER (ORDER BY COUNT(*) DESC, a) AS r FROM x GROUP BY a ORDER BY r"
        ),
        [(1, 1), (2, 2), (3, 3)]
    );
    assert_eq!(
        pairs(
            &db,
            "SELECT a, SUM(SUM(b)) OVER (PARTITION BY COUNT(*)) AS s FROM x GROUP BY a ORDER BY a"
        ),
        [(1, 50), (2, 50), (3, 50)]
    );
}
