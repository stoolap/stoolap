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

//! A DELETE names the rows it removes by primary key. A table without one
//! names them by the ids the scan read them under, so a condition the
//! storage layer cannot answer on its own still removes what it matched.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE a (id INTEGER, n INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER)", ()).unwrap();
    db.execute("INSERT INTO a VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (2)", ()).unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

#[test]
fn test_delete_with_an_in_subquery() {
    let db = setup("delete_no_pk_in");
    let removed = db
        .execute("DELETE FROM a WHERE id IN (SELECT id FROM b)", ())
        .unwrap();
    assert_eq!(removed, 1);
    assert_eq!(ids(&db, "SELECT id FROM a ORDER BY id"), [1, 3]);
}

#[test]
fn test_delete_with_exists_and_not_exists() {
    let db = setup("delete_no_pk_exists");
    let removed = db
        .execute(
            "DELETE FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.id = a.id)",
            (),
        )
        .unwrap();
    assert_eq!(removed, 1);
    assert_eq!(ids(&db, "SELECT id FROM a ORDER BY id"), [1, 3]);

    let removed = db
        .execute(
            "DELETE FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.id = a.id)",
            (),
        )
        .unwrap();
    assert_eq!(removed, 2);
    assert_eq!(ids(&db, "SELECT id FROM a ORDER BY id"), Vec::<i64>::new());
}

/// A condition the storage layer cannot answer on its own reaches the same
/// rows a SELECT would
#[test]
fn test_delete_with_a_condition_over_two_columns() {
    let db = setup("delete_no_pk_expression");
    let removed = db.execute("DELETE FROM a WHERE id + n > 21", ()).unwrap();
    assert_eq!(removed, 2);
    assert_eq!(ids(&db, "SELECT id FROM a ORDER BY id"), [1]);
}

/// The rows a DELETE returns are the rows it removed
#[test]
fn test_delete_returning() {
    let db = setup("delete_no_pk_returning");
    let returned: Vec<i64> = db
        .query(
            "DELETE FROM a WHERE id IN (SELECT id FROM b) RETURNING id",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    assert_eq!(returned, [2]);
    assert_eq!(ids(&db, "SELECT id FROM a ORDER BY id"), [1, 3]);
}

/// Two rows holding the same values are two rows
#[test]
fn test_rows_that_hold_the_same_values() {
    let db = Database::open("memory://delete_no_pk_twins").unwrap();
    db.execute("CREATE TABLE t (n INTEGER)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1), (1), (2)", ())
        .unwrap();
    let removed = db
        .execute("DELETE FROM t WHERE n IN (SELECT 1)", ())
        .unwrap();
    assert_eq!(removed, 2, "both rows holding a 1");
    assert_eq!(ids(&db, "SELECT n FROM t"), [2]);
}

/// A key that names a column of a table without a primary key is enforced
/// the same way, since the rows are still read before they go
#[test]
fn test_foreign_keys_on_a_parent_without_a_primary_key() {
    let db = Database::open("memory://delete_no_pk_restrict").unwrap();
    db.execute("CREATE TABLE par (code INTEGER UNIQUE, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE ch (id INTEGER PRIMARY KEY, code INTEGER REFERENCES par(code) ON DELETE RESTRICT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO par VALUES (1, 'a'), (2, 'b')", ())
        .unwrap();
    db.execute("INSERT INTO ch VALUES (1, 1)", ()).unwrap();

    assert!(
        db.execute("DELETE FROM par WHERE code IN (SELECT 1)", ())
            .is_err(),
        "the child still references it"
    );
    assert_eq!(ids(&db, "SELECT code FROM par ORDER BY code"), [1, 2]);
    assert_eq!(ids(&db, "SELECT id FROM ch"), [1]);

    let removed = db
        .execute("DELETE FROM par WHERE code IN (SELECT 2)", ())
        .unwrap();
    assert_eq!(removed, 1, "the row nothing references goes");
    assert_eq!(ids(&db, "SELECT code FROM par"), [1]);
}

#[test]
fn test_cascade_on_a_parent_without_a_primary_key() {
    let db = Database::open("memory://delete_no_pk_cascade").unwrap();
    db.execute("CREATE TABLE par (code INTEGER UNIQUE)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE ch (id INTEGER PRIMARY KEY, code INTEGER REFERENCES par(code) ON DELETE CASCADE)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO par VALUES (1), (2)", ()).unwrap();
    db.execute("INSERT INTO ch VALUES (1, 1), (2, 2)", ())
        .unwrap();

    db.execute("DELETE FROM par WHERE code IN (SELECT 1)", ())
        .unwrap();
    assert_eq!(ids(&db, "SELECT code FROM par"), [2]);
    assert_eq!(ids(&db, "SELECT id FROM ch"), [2], "the child went with it");
}
