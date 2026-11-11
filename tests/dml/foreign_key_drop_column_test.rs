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

//! A column a foreign key references does not drop; a foreign key's own
//! column drops with its key, live and after the database opens again

use stoolap::Database;

fn open(dir: &tempfile::TempDir) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap()
}

fn shown(db: &Database, table: &str) -> String {
    db.query(&format!("SHOW CREATE TABLE {table}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(1).unwrap())
        .collect()
}

#[test]
fn a_column_a_foreign_key_references_does_not_drop() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pk INTEGER REFERENCES p(k))",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, up INTEGER REFERENCES s(k))",
        "INSERT INTO p VALUES (1, 10)",
        "INSERT INTO c VALUES (100, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (sql, child) in [
        ("ALTER TABLE p DROP COLUMN k", "c(pk)"),
        ("ALTER TABLE s DROP COLUMN k", "s(up)"),
    ] {
        let message = db.execute(sql, ()).unwrap_err().to_string();
        assert!(
            message.contains(&format!("the foreign key on {child} references it")),
            "{sql}: {message}"
        );
    }
    assert!(shown(&db, "p").contains("k Integer"), "{}", shown(&db, "p"));
    assert!(
        db.execute("DELETE FROM p WHERE id = 1", ()).is_err(),
        "a referenced parent row was deleted"
    );
    db.execute("DROP TABLE c", ()).unwrap();
    db.execute("ALTER TABLE p DROP COLUMN k", ())
        .unwrap_or_else(|e| panic!("with no key referencing it: {e}"));
}

#[test]
fn a_foreign_key_column_drops_with_its_key() {
    for reopen in ["live", "log", "checkpoint"] {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open(&dir);
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id), x INTEGER)",
            "INSERT INTO p VALUES (1)",
            "INSERT INTO c VALUES (10, 1, 5)",
            "ALTER TABLE c DROP COLUMN pid",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if reopen != "live" {
            if reopen == "checkpoint" {
                db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            }
            db.close().unwrap();
            db = open(&dir);
        }
        let shown = shown(&db, "c");
        assert!(!shown.contains("FOREIGN KEY"), "{reopen}: {shown}");
        for sql in ["INSERT INTO c VALUES (11, 7)", "DELETE FROM p WHERE id = 1"] {
            db.execute(sql, ())
                .unwrap_or_else(|e| panic!("{reopen}: {sql}: {e}"));
        }
        db.close().unwrap();
    }
}
