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

fn rows(db: &Database, sql: &str) -> Vec<Vec<i64>> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len()).map(|i| row.get::<i64>(i).unwrap()).collect()
        })
        .collect()
}

/// Each layout the rows are read from: the hot store, then volumes
fn each_layout(name: &str, check: impl Fn(&Database)) {
    let tables = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY REFERENCES p(u) ON UPDATE CASCADE, x INTEGER)",
            "INSERT INTO p VALUES (1, 10), (2, 20), (3, 30)",
            "INSERT INTO c VALUES (10, 100), (20, 200)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&tables(
        &format!("memory://fk_cascade_into_key_{name}"),
        false,
    ));
    let dir = tempfile::tempdir().unwrap();
    check(&tables(&format!("file://{}", dir.path().display()), true));
}

fn refused(db: &Database, sql: &str) {
    let error = db.execute(sql, ()).unwrap_err().to_string();
    assert!(
        error.contains("into the primary key of table"),
        "{sql}: {error}"
    );
}

#[test]
fn a_cascade_that_would_change_a_child_key_is_refused_before_any_write() {
    each_layout("refused", |db| {
        refused(db, "UPDATE p SET u = 40 WHERE id = 1");
        refused(db, "UPDATE p SET u = u + 100");
        let unchanged = vec![vec![1, 10], vec![2, 20], vec![3, 30]];
        assert_eq!(rows(db, "SELECT id, u FROM p ORDER BY id"), unchanged);
        assert_eq!(
            rows(db, "SELECT id, x FROM c ORDER BY id"),
            [[10, 100], [20, 200]]
        );
        // Each child is still found by its key, and its key stays unique
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 10"), [[100]]);
        assert!(db.execute("INSERT INTO c VALUES (20, 9)", ()).is_err());
    });
}

#[test]
fn a_parent_value_no_child_names_is_updated() {
    each_layout("unreferenced", |db| {
        db.execute("UPDATE p SET u = 31 WHERE id = 3", ()).unwrap();
        assert_eq!(rows(db, "SELECT u FROM p WHERE id = 3"), [[31]]);
    });
}

#[test]
fn a_refused_cascade_leaves_an_open_transaction_as_it_was() {
    each_layout("txn", |db| {
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE p SET u = 33 WHERE id = 3", ()).unwrap();
        refused(db, "UPDATE p SET u = 40 WHERE id = 1");
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(
            rows(db, "SELECT id, u FROM p ORDER BY id"),
            [[1, 10], [2, 20], [3, 33]]
        );
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 10"), [[100]]);
    });
}

#[test]
fn a_cascade_reaching_a_grandchild_key_is_refused_before_any_write() {
    let db = Database::open("memory://fk_cascade_into_key_grandchild").unwrap();
    for sql in [
        "CREATE TABLE g (id INTEGER PRIMARY KEY, a INTEGER UNIQUE)",
        "CREATE TABLE h (k INTEGER PRIMARY KEY, ga INTEGER UNIQUE REFERENCES g(a) ON UPDATE CASCADE)",
        "CREATE TABLE i (z INTEGER PRIMARY KEY REFERENCES h(ga) ON UPDATE CASCADE)",
        "INSERT INTO g VALUES (1, 5)",
        "INSERT INTO h VALUES (1, 5)",
        "INSERT INTO i VALUES (5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    refused(&db, "UPDATE g SET a = 6 WHERE id = 1");
    assert_eq!(rows(&db, "SELECT a FROM g"), [[5]]);
    assert_eq!(rows(&db, "SELECT ga FROM h"), [[5]]);
    assert_eq!(rows(&db, "SELECT z FROM i WHERE z = 5"), [[5]]);
}
