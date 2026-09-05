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

//! A subquery whose only tie to the parent row sits in a subquery of its
//! own is still correlated: it is run once per parent row, in WHERE as it
//! already was in the select list.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE nx (id INTEGER PRIMARY KEY, b INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE ny (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO nx VALUES (1, 5, 'p'), (2, 9, 'q'), (3, 1, 'p'), (4, 4, 'q'), (5, 7, 'r')",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO ny VALUES (1, 2, 10, 'p'), (2, 5, 2, 'q'), (3, 3, 6, 'p'), (4, 2, 8, 'r'), (5, NULL, 1, 'q')",
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
fn test_a_scalar_subquery_tied_to_the_parent_through_its_own_subquery() {
    let db = setup("nested_outer_scalar");
    // the least a among ny rows sharing the parent's s: p and r give 2, q gives 5;
    // the mean b at a = 2 is 9, at a = 5 it is 2
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM nx WHERE b > (SELECT AVG(b) FROM ny WHERE ny.a = (SELECT MIN(a) FROM ny WHERE ny.s = nx.s)) ORDER BY id"
        ),
        [2, 4]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM nx WHERE b > (SELECT AVG(b) FROM ny WHERE ny.a IN (SELECT MIN(z.a) FROM ny z WHERE z.s = nx.s)) ORDER BY id"
        ),
        [2, 4]
    );
}

#[test]
fn test_an_exists_tied_to_the_parent_through_its_own_subquery() {
    let db = setup("nested_outer_exists");
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM nx WHERE EXISTS (SELECT 1 FROM ny WHERE ny.a = (SELECT MIN(a) FROM ny WHERE ny.s = nx.s) AND ny.b > 5) ORDER BY id"
        ),
        [1, 3, 5]
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM nx WHERE NOT EXISTS (SELECT 1 FROM ny WHERE ny.a = (SELECT MIN(a) FROM ny WHERE ny.s = nx.s) AND ny.b > 5) ORDER BY id"
        ),
        [2, 4]
    );
}
