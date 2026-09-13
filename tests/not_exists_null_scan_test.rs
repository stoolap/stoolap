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

//! The NULL handling of a NOT EXISTS answered through a semi-join set,
//! now decided by one set lookup instead of a walk over the set.

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn a_null_inner_value_matches_no_parent() {
    let db = Database::open("memory://not_exists_null_inner").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2), (3)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1), (2, NULL)", ())
        .unwrap();
    for _ in 0..3 {
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM p WHERE NOT EXISTS (SELECT 1 FROM c WHERE c.p_id = p.id) ORDER BY id"
            ),
            vec![2, 3]
        );
    }
}

#[test]
fn a_null_outer_value_is_kept_by_not_exists() {
    let db = Database::open("memory://not_exists_null_outer").unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, ref_id INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, p_id INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1, 10), (2, NULL), (3, 30)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 10), (2, 20)", ())
        .unwrap();
    // No child can equal a NULL reference, so row 2 has no child
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE NOT EXISTS (SELECT 1 FROM c WHERE c.p_id = p.ref_id) ORDER BY id"
        ),
        vec![2, 3]
    );
}

#[test]
fn a_negated_literal_list_holding_null_keeps_nothing() {
    let db = Database::open("memory://not_exists_null_literal").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO p VALUES (1), (2), (3)", ())
        .unwrap();
    // Nothing is known to be outside a list that holds a NULL
    assert!(ids(&db, "SELECT id FROM p WHERE id NOT IN (1, NULL)").is_empty());
    assert!(ids(
        &db,
        "SELECT id FROM p WHERE id NOT IN (1, NULL) OR id IS NULL ORDER BY id"
    )
    .is_empty());
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM p WHERE id NOT IN (1) OR id IS NULL ORDER BY id"
        ),
        vec![2, 3]
    );
}
