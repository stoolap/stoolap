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

/// A key that cascades and a key that refuses, in that order, must leave
/// nothing behind when the refusal comes
#[test]
fn test_a_refusal_leaves_no_cascade_behind() {
    for parent in [
        "CREATE TABLE par (code INTEGER UNIQUE)",
        "CREATE TABLE par (code INTEGER PRIMARY KEY)",
    ] {
        let keyed = parent.contains("PRIMARY KEY");
        let db = Database::open(&format!(
            "memory://delete_fk_atomic_{}",
            if keyed { "pk" } else { "nopk" }
        ))
        .unwrap();
        db.execute(parent, ()).unwrap();
        db.execute(
            "CREATE TABLE kid (id INTEGER PRIMARY KEY, a INTEGER REFERENCES par(code) ON DELETE CASCADE, b INTEGER REFERENCES par(code) ON DELETE RESTRICT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO par VALUES (1), (2)", ()).unwrap();
        db.execute("INSERT INTO kid VALUES (1, 1, NULL), (2, NULL, 1)", ())
            .unwrap();

        assert!(
            db.execute("DELETE FROM par WHERE code IN (SELECT 1)", ())
                .is_err(),
            "the second key refuses ({parent})"
        );
        assert_eq!(
            ids(&db, "SELECT code FROM par ORDER BY code"),
            [1, 2],
            "{parent}"
        );
        assert_eq!(
            ids(&db, "SELECT id FROM kid ORDER BY id"),
            [1, 2],
            "the first key cascaded nothing away ({parent})"
        );
    }
}

/// A key that refuses two levels down refuses the whole statement, before
/// a key one level down has taken anything away
#[test]
fn test_a_refusal_below_a_cascade_leaves_nothing_behind() {
    for parent in [
        "CREATE TABLE par (code INTEGER UNIQUE)",
        "CREATE TABLE par (code INTEGER PRIMARY KEY)",
    ] {
        let keyed = parent.contains("PRIMARY KEY");
        let db = Database::open(&format!(
            "memory://delete_fk_chain_{}",
            if keyed { "pk" } else { "nopk" }
        ))
        .unwrap();
        db.execute(parent, ()).unwrap();
        db.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, pcode INTEGER REFERENCES par(code) ON DELETE CASCADE)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE grand (id INTEGER PRIMARY KEY, cid INTEGER REFERENCES child(id) ON DELETE RESTRICT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO par VALUES (1), (2)", ()).unwrap();
        db.execute("INSERT INTO child VALUES (1, 1), (2, 2)", ())
            .unwrap();
        db.execute("INSERT INTO grand VALUES (1, 2)", ()).unwrap();

        db.execute("BEGIN", ()).unwrap();
        assert!(
            db.execute("DELETE FROM par WHERE code <= 2", ()).is_err(),
            "the grandchild refuses ({parent})"
        );
        assert_eq!(
            ids(&db, "SELECT id FROM child ORDER BY id"),
            [1, 2],
            "the first parent's child did not go before the refusal ({parent})"
        );
        assert_eq!(
            ids(&db, "SELECT code FROM par ORDER BY code"),
            [1, 2],
            "{parent}"
        );
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(
            ids(&db, "SELECT id FROM child ORDER BY id"),
            [1, 2],
            "nothing was left to commit ({parent})"
        );
    }
}

/// A key beneath a cascade names the column it points at, which need not
/// be the child's primary key; the walk reads the child rows through that
/// column, before anything goes and while the cascade runs
#[test]
fn test_keys_beneath_a_cascade_read_the_column_they_name() {
    for (parent, child) in [
        (
            "CREATE TABLE par (code INTEGER UNIQUE)",
            "CREATE TABLE child (ccode INTEGER UNIQUE, pcode INTEGER REFERENCES par(code) ON DELETE CASCADE)",
        ),
        (
            "CREATE TABLE par (code INTEGER PRIMARY KEY)",
            "CREATE TABLE child (id INTEGER PRIMARY KEY, ccode INTEGER UNIQUE, pcode INTEGER REFERENCES par(code) ON DELETE CASCADE)",
        ),
    ] {
        let keyed = child.contains("PRIMARY KEY");
        let db = Database::open(&format!(
            "memory://delete_fk_unique_chain_{}",
            if keyed { "keyed" } else { "unique" }
        ))
        .unwrap();
        db.execute(parent, ()).unwrap();
        db.execute(child, ()).unwrap();
        db.execute(
            "CREATE TABLE grand (id INTEGER PRIMARY KEY, ccode INTEGER REFERENCES child(ccode) ON DELETE RESTRICT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO par VALUES (1), (2)", ()).unwrap();
        if keyed {
            db.execute("INSERT INTO child VALUES (1, 10, 1), (2, 20, 2)", ())
                .unwrap();
        } else {
            db.execute("INSERT INTO child VALUES (10, 1), (20, 2)", ())
                .unwrap();
        }
        db.execute("INSERT INTO grand VALUES (1, 20)", ()).unwrap();

        // The grandchild holds child 20, which belongs to parent 2
        assert!(
            db.execute("DELETE FROM par WHERE code = 2", ()).is_err(),
            "the grandchild refuses ({child})"
        );
        assert_eq!(ids(&db, "SELECT ccode FROM child ORDER BY ccode"), [10, 20], "{child}");
        assert_eq!(ids(&db, "SELECT id FROM grand"), [1], "{child}");

        // Parent 1's child has nothing beneath it and goes with the parent
        db.execute("DELETE FROM par WHERE code = 1", ()).unwrap();
        assert_eq!(ids(&db, "SELECT ccode FROM child"), [20], "{child}");
        assert_eq!(ids(&db, "SELECT code FROM par"), [2], "{child}");
    }
}
