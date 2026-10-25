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

//! A foreign key finds every row whose key equals the other side's value,
//! whatever the two numeric types and whether the row is hot or sealed

use stoolap::Database;

const TYPES: [(&str, &str); 4] = [
    ("INTEGER", "FLOAT"),
    ("FLOAT", "INTEGER"),
    ("INTEGER", "INTEGER"),
    ("FLOAT", "FLOAT"),
];

fn open(dir: &tempfile::TempDir) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap()
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

/// A parent `(1, 1)` and a child `(1, 1)` under `action`, the child sealed
/// with `sealed`
fn setup(parent: &str, child: &str, action: &str, sealed: bool) -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        format!("CREATE TABLE p (id INTEGER PRIMARY KEY, k {parent} UNIQUE)"),
        format!("CREATE TABLE c (id INTEGER PRIMARY KEY, k {child} REFERENCES p(k) {action})"),
        "INSERT INTO p VALUES (1, 1)".to_string(),
        "INSERT INTO c VALUES (1, 1)".to_string(),
    ] {
        db.execute(&sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    (dir, db)
}

#[test]
fn a_parent_change_finds_its_children_in_both_layers_whatever_the_types() {
    let mut wrong = Vec::new();
    for (parent, child) in TYPES {
        for sealed in [false, true] {
            let layer = if sealed { "sealed" } else { "hot" };
            for (action, statement, refused, children_after) in [
                (
                    "ON UPDATE CASCADE",
                    "UPDATE p SET k = 3 WHERE id = 1",
                    false,
                    "SELECT COUNT(*) FROM c WHERE k = 3",
                ),
                (
                    "ON UPDATE RESTRICT",
                    "UPDATE p SET k = 3 WHERE id = 1",
                    true,
                    "SELECT COUNT(*) FROM c WHERE k = 1",
                ),
                (
                    "ON DELETE RESTRICT",
                    "DELETE FROM p WHERE id = 1",
                    true,
                    "SELECT COUNT(*) FROM c WHERE k = 1",
                ),
                (
                    "ON DELETE CASCADE",
                    "DELETE FROM p WHERE id = 1",
                    false,
                    "SELECT 1 - COUNT(*) FROM c",
                ),
                (
                    "ON DELETE SET NULL",
                    "DELETE FROM p WHERE id = 1",
                    false,
                    "SELECT COUNT(*) FROM c WHERE k IS NULL",
                ),
            ] {
                let (_dir, db) = setup(parent, child, action, sealed);
                let result = db.execute(statement, ());
                let state = count(&db, children_after);
                if result.is_err() != refused || state != 1 {
                    wrong.push(format!(
                        "{parent}/{child} {layer} {action}: {statement} -> {result:?}, \
                         {children_after} = {state}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_child_finds_its_parent_in_both_layers_whatever_the_types() {
    let mut wrong = Vec::new();
    for (parent, child) in TYPES {
        for sealed in [false, true] {
            let layer = if sealed { "sealed" } else { "hot" };
            let (_dir, db) = setup(parent, child, "ON DELETE RESTRICT", sealed);
            for (statement, refused) in [
                ("INSERT INTO c VALUES (2, 1)", false),
                ("INSERT INTO c VALUES (3, 2)", true),
                ("UPDATE c SET k = 1 WHERE id = 1", false),
                ("UPDATE c SET k = 4 WHERE id = 1", true),
            ] {
                let result = db.execute(statement, ());
                if result.is_err() != refused {
                    wrong.push(format!(
                        "{parent}/{child} {layer}: {statement} -> {result:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_cascade_finds_the_grandchildren_of_a_sealed_child() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir);
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, k FLOAT REFERENCES p(k) ON DELETE CASCADE, \
             ck INTEGER UNIQUE)",
            "CREATE TABLE g (id INTEGER PRIMARY KEY, ck FLOAT REFERENCES c(ck) ON DELETE RESTRICT)",
            "INSERT INTO p VALUES (1, 1)",
            "INSERT INTO c VALUES (1, 1, 5)",
            "INSERT INTO g VALUES (1, 5)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        let result = db.execute("DELETE FROM p WHERE id = 1", ());
        assert!(
            result.is_err(),
            "sealed {sealed}: the grandchild restricts the cascade, got {result:?}"
        );
        assert_eq!(count(&db, "SELECT COUNT(*) FROM c"), 1, "sealed {sealed}");
    }
}
