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

//! OVER (w ...) builds on a named window: it keeps the window's partition
//! and takes its own ORDER BY, where the window has none, and its own frame.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE hx (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO hx VALUES (1, 1, 10), (2, 1, 11), (3, 2, 50), (4, 3, 30), (5, 3, 40), (6, 3, 5)",
        (),
    )
    .unwrap();
    db
}

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
fn test_a_frame_on_top_of_a_named_window() {
    let db = setup("named_window_frame");
    assert_eq!(
        pairs(
            &db,
            "SELECT id, SUM(b) OVER (w ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM hx WINDOW w AS (PARTITION BY a ORDER BY id) ORDER BY id"
        ),
        [(1, 10), (2, 21), (3, 50), (4, 30), (5, 70), (6, 45)]
    );
    assert_eq!(
        pairs(
            &db,
            "SELECT id, SUM(b) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM hx WINDOW w AS (PARTITION BY a ORDER BY id) ORDER BY id"
        ),
        [(1, 21), (2, 21), (3, 50), (4, 75), (5, 75), (6, 75)]
    );
}

#[test]
fn test_an_order_by_on_top_of_a_partition_only_window() {
    let db = setup("named_window_order");
    assert_eq!(
        pairs(
            &db,
            "SELECT id, RANK() OVER (w ORDER BY b DESC) FROM hx WINDOW w AS (PARTITION BY a) ORDER BY id"
        ),
        [(1, 2), (2, 1), (3, 1), (4, 2), (5, 1), (6, 3)]
    );
    assert_eq!(
        pairs(
            &db,
            "SELECT id, COUNT(*) OVER (w) FROM hx WINDOW w AS (PARTITION BY a) ORDER BY id"
        ),
        [(1, 2), (2, 2), (3, 1), (4, 3), (5, 3), (6, 3)]
    );
}
