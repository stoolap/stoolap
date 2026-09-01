// Copyright 2026 Stoolap Contributors
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

//! ORDER BY on a grouped column must hold even when that column is not
//! in the SELECT list.

use stoolap::Database;

fn setup(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 7, 'b'), (2, 1, 'a'), (3, 7, 'b'), (4, 5, 'c')",
        (),
    )
    .unwrap();
    db
}

#[test]
fn order_by_unprojected_group_column() {
    let db = setup("memory://gb_order_hidden");
    let sums: Vec<i64> = db
        .query("SELECT SUM(id) FROM t GROUP BY v ORDER BY v", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    // groups v=1 (id 2), v=5 (id 4), v=7 (ids 1+3)
    assert_eq!(sums, vec![2, 4, 4], "ascending by the grouped column");

    let sums: Vec<i64> = db
        .query("SELECT SUM(id) FROM t GROUP BY v ORDER BY v DESC", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    assert_eq!(sums, vec![4, 4, 2], "descending by the grouped column");
}

#[test]
fn order_by_unprojected_group_column_text() {
    let db = setup("memory://gb_order_hidden_text");
    let counts: Vec<i64> = db
        .query("SELECT COUNT(*) FROM t GROUP BY w ORDER BY w DESC", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    // groups 'c' (1), 'b' (2), 'a' (1)
    assert_eq!(counts, vec![1, 2, 1]);
}

#[test]
fn order_by_projected_group_column_still_works() {
    let db = setup("memory://gb_order_projected");
    let vs: Vec<i64> = db
        .query("SELECT v, SUM(id) FROM t GROUP BY v ORDER BY v", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    assert_eq!(vs, vec![1, 5, 7]);
}
