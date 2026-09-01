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

#[test]
fn order_by_unprojected_qualified_group_column() {
    let db = setup("memory://gb_order_qualified");
    let sums: Vec<i64> = db
        .query("SELECT SUM(t.id) FROM t GROUP BY t.v ORDER BY t.v", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    assert_eq!(sums, vec![2, 4, 4], "qualified group key must still sort");

    let sums: Vec<i64> = db
        .query("SELECT SUM(id) FROM t GROUP BY v ORDER BY t.v DESC", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    assert_eq!(sums, vec![4, 4, 2], "qualified ORDER BY over bare GROUP BY");
}

#[test]
fn order_by_unprojected_group_column_with_window() {
    let db = setup("memory://gb_order_window");
    let mins: Vec<i64> = db
        .query(
            "SELECT MIN(id), ROW_NUMBER() OVER () FROM t GROUP BY v ORDER BY v",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    // groups v=1 (min id 2), v=5 (4), v=7 (1)
    assert_eq!(
        mins,
        vec![2, 4, 1],
        "window projection must keep the sort key"
    );
}

#[test]
fn bare_order_by_over_qualified_group_key() {
    let db = setup("memory://gb_order_bare_over_qualified");
    let maxs: Vec<i64> = db
        .query("SELECT MAX(id) FROM t GROUP BY t.v ORDER BY v", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    // groups v=1 (max id 2), v=5 (4), v=7 (3)
    assert_eq!(
        maxs,
        vec![2, 4, 3],
        "bare ORDER BY must find the qualified key"
    );
}

#[test]
fn qualified_order_by_prefers_group_key_over_alias() {
    let db = setup("memory://gb_order_alias_shadow");
    let vals: Vec<i64> = db
        .query("SELECT MAX(id) AS v FROM t GROUP BY t.v ORDER BY t.v", ())
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    // ordering by the group key t.v (1, 5, 7) yields max ids 2, 4, 3;
    // ordering by the alias MAX(id) would yield 2, 3, 4
    assert_eq!(vals, vec![2, 4, 3], "qualified key must beat the alias");
}

#[test]
fn bare_order_by_over_qualified_group_key_with_window() {
    let db = setup("memory://gb_order_bare_window");
    let maxs: Vec<i64> = db
        .query(
            "SELECT MAX(id), ROW_NUMBER() OVER () FROM t GROUP BY t.v ORDER BY v",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap())
        .collect();
    assert_eq!(maxs, vec![2, 4, 3]);
}
