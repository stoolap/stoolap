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

use stoolap::Database;

/// Outer pairs (1, 10), (1, 20), (2, 10); each outer row's own inner pair
/// is (1, 10), (1, 20), (2, 99), so the first two match and the third has
/// its first column right and its second wrong
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 10), (2, 1, 20), (3, 2, 10)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER, y INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO u VALUES (1, 1, 1, 10), (2, 2, 1, 20), (3, 3, 2, 99)",
        (),
    )
    .unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn test_a_correlated_tuple_in_compares_every_column() {
    let db = setup("correlated_tuple_in");
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE (a, b) IN (SELECT x, y FROM u WHERE u.k = t.id) ORDER BY id"
        ),
        vec![1, 2],
        "row 3 matches on its first column only"
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE (a, b) NOT IN (SELECT x, y FROM u WHERE u.k = t.id) ORDER BY id"
        ),
        vec![3]
    );
}

#[test]
fn test_an_uncorrelated_tuple_in_keeps_its_answer() {
    let db = setup("uncorrelated_tuple_in");
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM t WHERE (a, b) IN (SELECT x, y FROM u) ORDER BY id"
        ),
        vec![1, 2]
    );
}
