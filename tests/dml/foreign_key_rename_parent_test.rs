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

//! A foreign key follows its parent table through a rename, live and after
//! the database opens again from its log or from a checkpoint

use stoolap::Database;

fn open(dir: &tempfile::TempDir) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap()
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one(sql, ()).unwrap()
}

#[test]
fn a_child_key_follows_its_renamed_parent() {
    for reopen in ["live", "log", "checkpoint"] {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open(&dir);
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id))",
            "INSERT INTO p VALUES (1), (2)",
            "INSERT INTO c VALUES (10, 1)",
            "ALTER TABLE p RENAME TO p2",
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
        db.execute("INSERT INTO c VALUES (11, 2)", ())
            .unwrap_or_else(|e| panic!("{reopen}: child insert: {e}"));
        for refused in [
            "INSERT INTO c VALUES (12, 3)",
            "DELETE FROM p2 WHERE id = 1",
            "DROP TABLE p2",
        ] {
            assert!(
                db.execute(refused, ()).is_err(),
                "{reopen}: {refused} must be refused"
            );
        }
        assert_eq!(count(&db, "SELECT COUNT(*) FROM p2"), 2, "{reopen}");
        assert_eq!(count(&db, "SELECT COUNT(*) FROM c"), 2, "{reopen}");
        db.close().unwrap();
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_failed_rename_leaves_no_child_lookup_of_the_new_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id))",
        "CREATE TABLE u (id INTEGER PRIMARY KEY)",
        "INSERT INTO p VALUES (1)",
        "INSERT INTO c VALUES (10, 1)",
        "PRAGMA CHECKPOINT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // A non-empty directory in the way fails the volume directory's move
    let blocker = dir.path().join("volumes").join("p2");
    std::fs::create_dir_all(&blocker).unwrap();
    std::fs::write(blocker.join("x"), b"x").unwrap();
    let other = db.clone();
    stoolap::test_failpoints::in_rename_before_the_move(move || {
        other.execute("DELETE FROM u WHERE id = 1", ()).unwrap();
    });
    assert!(db.execute("ALTER TABLE p RENAME TO p2", ()).is_err());
    assert!(
        db.execute("DELETE FROM p WHERE id = 1", ()).is_err(),
        "c still references p's row 1"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM p"), 1);
}
