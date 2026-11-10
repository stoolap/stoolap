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

//! A foreign key follows its parent table and the columns it names through
//! a rename, live and after the database opens again from its log or from a
//! checkpoint

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

#[test]
fn a_key_follows_its_renamed_columns() {
    for reopen in ["live", "log", "checkpoint"] {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open(&dir);
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, \
             pk INTEGER UNIQUE REFERENCES p(k) ON UPDATE CASCADE)",
            "CREATE TABLE g (id INTEGER PRIMARY KEY, cpk INTEGER REFERENCES c(pk) ON UPDATE CASCADE)",
            "INSERT INTO p VALUES (1, 10), (2, 30)",
            "INSERT INTO c VALUES (100, 10)",
            "INSERT INTO g VALUES (1000, 10)",
            "ALTER TABLE p RENAME COLUMN k TO k2",
            "ALTER TABLE c RENAME COLUMN pk TO pk2",
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
        let shown: String = db
            .query("SHOW CREATE TABLE c", ())
            .unwrap()
            .map(|row| row.unwrap().get::<String>(1).unwrap())
            .collect();
        assert!(
            shown.contains("FOREIGN KEY (pk2) REFERENCES p(k2)"),
            "{reopen}: {shown}"
        );
        db.execute("INSERT INTO c VALUES (101, 30)", ())
            .unwrap_or_else(|e| panic!("{reopen}: child insert: {e}"));
        for refused in [
            "INSERT INTO c VALUES (102, 99)",
            "DELETE FROM p WHERE id = 1",
        ] {
            assert!(
                db.execute(refused, ()).is_err(),
                "{reopen}: {refused} must be refused"
            );
        }
        db.execute("UPDATE p SET k2 = 20 WHERE id = 1", ())
            .unwrap_or_else(|e| panic!("{reopen}: parent update: {e}"));
        assert_eq!(
            count(&db, "SELECT COUNT(*) FROM c WHERE pk2 = 20"),
            1,
            "{reopen}: the child follows the cascade"
        );
        assert_eq!(
            count(&db, "SELECT COUNT(*) FROM g WHERE cpk = 20"),
            1,
            "{reopen}: the grandchild follows the child's renamed column"
        );
        db.close().unwrap();
    }
}

#[test]
fn a_key_held_by_an_open_delete_stays_held_through_a_rename() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pk INTEGER REFERENCES p(k))",
        "INSERT INTO p VALUES (1, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut deleting = db.begin().unwrap();
    deleting.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    let other = db.clone();
    other
        .execute("ALTER TABLE p RENAME COLUMN k TO k2", ())
        .unwrap();
    let inserted = other.execute("INSERT INTO c VALUES (1, 10)", ());
    deleting.commit().unwrap();
    assert!(
        inserted.is_err(),
        "a child of the key the open delete removes was inserted"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c"), 0);
}

#[test]
fn a_key_held_stays_held_when_its_column_takes_a_dropped_column_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, j INTEGER UNIQUE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pk INTEGER REFERENCES p(k))",
        "CREATE TABLE d (id INTEGER PRIMARY KEY, pj INTEGER REFERENCES p(j))",
        "INSERT INTO p VALUES (1, 10, 20)",
        // A key of k was held once, and k is gone before the delete
        "INSERT INTO c VALUES (1, 10)",
        "DROP TABLE c",
        "ALTER TABLE p DROP COLUMN k",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut deleting = db.begin().unwrap();
    deleting.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    let other = db.clone();
    other
        .execute("ALTER TABLE p RENAME COLUMN j TO k", ())
        .unwrap();
    let inserted = other.execute("INSERT INTO d VALUES (1, 20)", ());
    deleting.commit().unwrap();
    assert!(
        inserted.is_err(),
        "a child of the key the open delete removes was inserted"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM d"), 0);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_key_held_stays_held_for_a_check_that_read_the_old_name() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER UNIQUE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pk INTEGER REFERENCES p(k))",
        "INSERT INTO p VALUES (1, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut deleting = db.begin().unwrap();
    deleting.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    let renaming = db.clone();
    stoolap::test_failpoints::before_fk_parent_hold(move || {
        renaming
            .execute("ALTER TABLE p RENAME COLUMN k TO k2", ())
            .unwrap();
    });
    let inserted = db.clone().execute("INSERT INTO c VALUES (1, 10)", ());
    deleting.commit().unwrap();
    assert!(
        inserted.is_err(),
        "a child of the key the open delete removes was inserted"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM c"), 0);
}

#[test]
fn a_key_held_stays_held_when_a_dropped_column_name_returns_in_another_case() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, \"K\" INTEGER UNIQUE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pk INTEGER REFERENCES p(\"K\"))",
        "INSERT INTO p VALUES (1, 10)",
        "INSERT INTO c VALUES (1, 10)",
        "DROP TABLE c",
        "ALTER TABLE p DROP COLUMN \"K\"",
        "ALTER TABLE p ADD COLUMN k INTEGER",
        "CREATE UNIQUE INDEX p_k ON p(k)",
        "UPDATE p SET k = 10",
        "CREATE TABLE d (id INTEGER PRIMARY KEY, pk INTEGER REFERENCES p(k))",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut deleting = db.begin().unwrap();
    deleting.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    let other = db.clone();
    other
        .execute("ALTER TABLE p RENAME COLUMN k TO k2", ())
        .unwrap();
    let inserted = other.execute("INSERT INTO d VALUES (1, 10)", ());
    deleting.commit().unwrap();
    assert!(
        inserted.is_err(),
        "a child of the key the open delete removes was inserted"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM d"), 0);
}

#[test]
fn a_shared_key_hold_moves_down_with_its_column_across_a_drop() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER, j INTEGER UNIQUE)",
        "CREATE TABLE d (id INTEGER PRIMARY KEY, pj INTEGER REFERENCES p(j))",
        "INSERT INTO p VALUES (1, 10, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut inserting = db.begin().unwrap();
    inserting
        .execute("INSERT INTO d VALUES (1, 20)", ())
        .unwrap();
    let other = db.clone();
    other.execute("ALTER TABLE p DROP COLUMN k", ()).unwrap();
    let deleted = other.execute("DELETE FROM p WHERE id = 1", ());
    inserting.commit().unwrap();
    assert!(
        deleted.is_err(),
        "the parent of the child an open insert added was deleted"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM p"), 1);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_check_holds_its_key_where_a_drop_moved_the_column() {
    use stoolap::IsolationLevel::{ReadCommitted, SnapshotIsolation};
    for isolation in [ReadCommitted, SnapshotIsolation] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir);
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY, k INTEGER, j INTEGER UNIQUE)",
            "CREATE TABLE d (id INTEGER PRIMARY KEY, pj INTEGER REFERENCES p(j))",
            "INSERT INTO p VALUES (1, 10, 20)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        let mut inserting = db.clone().begin_with_isolation(isolation).unwrap();
        let dropping = db.clone();
        stoolap::test_failpoints::before_fk_parent_hold(move || {
            dropping.execute("ALTER TABLE p DROP COLUMN k", ()).unwrap();
        });
        let inserted = inserting.execute("INSERT INTO d VALUES (1, 20)", ());
        let _ = db.clone().execute("DELETE FROM p WHERE id = 1", ());
        let committed = inserting.commit();
        assert_eq!(
            count(
                &db,
                "SELECT COUNT(*) FROM d WHERE pj NOT IN (SELECT j FROM p)"
            ),
            0,
            "{isolation:?}: insert {inserted:?}, commit {committed:?}: a child without its parent"
        );
    }
}
