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

//! A child row's key check and a delete or key change of its parent in two
//! open transactions: one of them fails, and no child is left without its parent

use stoolap::{Database, IsolationLevel};

fn open(name: &str, on_delete: &str) -> Database {
    create(
        Database::open(&format!("memory://{name}")).unwrap(),
        on_delete,
    )
}

fn create(db: Database, on_delete: &str) -> Database {
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, code TEXT UNIQUE)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER, \
             FOREIGN KEY (pid) REFERENCES p (id) ON DELETE {on_delete})"
        ),
        (),
    )
    .unwrap();
    db.execute("INSERT INTO p VALUES (1, 'a'), (2, 'b')", ())
        .unwrap();
    db
}

fn orphans(db: &Database) -> i64 {
    db.query_one(
        "SELECT COUNT(*) FROM c WHERE pid IS NOT NULL AND pid NOT IN (SELECT id FROM p)",
        (),
    )
    .unwrap()
}

fn is_conflict(error: &stoolap::Error) -> bool {
    error.to_string().contains("write conflict")
}

#[test]
fn a_child_insert_against_an_uncommitted_parent_delete_is_a_write_conflict() {
    let db = open("fk_conc_child_after_delete", "RESTRICT");
    let other = db.clone();
    let mut deleting = other.begin().unwrap();
    deleting.execute("DELETE FROM p WHERE id = 1", ()).unwrap();

    let error = db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap_err();
    assert!(is_conflict(&error), "the insert failed with {error}");
    deleting.commit().unwrap();

    assert_eq!(orphans(&db), 0);
    let error = db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap_err();
    assert!(
        !is_conflict(&error),
        "the delete's hold outlived its commit: {error}"
    );
}

#[test]
fn a_parent_delete_against_an_uncommitted_child_insert_is_a_write_conflict() {
    for on_delete in ["RESTRICT", "CASCADE", "SET NULL"] {
        let db = open(
            &format!("fk_conc_delete_after_child_{}", on_delete.replace(' ', "_")),
            on_delete,
        );
        let other = db.clone();
        let mut inserting = other.begin().unwrap();
        inserting
            .execute("INSERT INTO c VALUES (1, 1)", ())
            .unwrap();

        let error = db.execute("DELETE FROM p WHERE id = 1", ()).unwrap_err();
        assert!(
            is_conflict(&error),
            "{on_delete}: the delete failed with {error}"
        );
        inserting.commit().unwrap();

        assert_eq!(orphans(&db), 0, "{on_delete}");
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM p WHERE id = 1", ())
                .unwrap(),
            1,
            "{on_delete}"
        );
    }
}

#[test]
fn a_key_change_against_an_uncommitted_child_insert_is_a_write_conflict() {
    let db = Database::open("memory://fk_conc_key_change").unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, code TEXT UNIQUE)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, code TEXT, \
         FOREIGN KEY (code) REFERENCES p (code))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO p VALUES (1, 'a')", ()).unwrap();
    let other = db.clone();
    let mut inserting = other.begin().unwrap();
    inserting
        .execute("INSERT INTO c VALUES (1, 'a')", ())
        .unwrap();

    let error = db
        .execute("UPDATE p SET code = 'z' WHERE id = 1", ())
        .unwrap_err();
    assert!(is_conflict(&error), "the update failed with {error}");
    inserting.commit().unwrap();

    assert_eq!(
        db.query_one::<i64, _>(
            "SELECT COUNT(*) FROM c WHERE code NOT IN (SELECT code FROM p)",
            ()
        )
        .unwrap(),
        0
    );
}

#[test]
fn children_of_one_parent_insert_side_by_side() {
    let db = open("fk_conc_side_by_side", "RESTRICT");
    let other = db.clone();
    let mut first = other.begin().unwrap();
    first.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap();

    db.execute("INSERT INTO c VALUES (2, 1)", ()).unwrap();
    first.commit().unwrap();

    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM c WHERE pid = 1", ())
            .unwrap(),
        2
    );
}

#[test]
fn a_rolled_back_delete_lets_its_key_go() {
    let db = open("fk_conc_rolled_back", "RESTRICT");
    let other = db.clone();
    let mut deleting = other.begin().unwrap();
    deleting.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    deleting.rollback().unwrap();

    db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap();
    assert_eq!(orphans(&db), 0);
}

#[test]
fn a_snapshot_child_insert_sees_a_sealed_parent_deleted_after_it_began() {
    let dir = tempfile::tempdir().unwrap();
    let db = create(
        Database::open(&format!(
            "file://{}?checkpoint_interval=0",
            dir.path().display()
        ))
        .unwrap(),
        "RESTRICT",
    );
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert!(!db.engine().volume_stats().is_empty());
    let other = db.clone();
    let mut inserting = other
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert_eq!(
        inserting
            .query_one::<i64, _>("SELECT COUNT(*) FROM p", ())
            .unwrap(),
        2
    );
    db.execute("DELETE FROM p WHERE id = 1", ()).unwrap();

    let error = inserting
        .execute("INSERT INTO c VALUES (1, 1)", ())
        .unwrap_err();
    assert!(
        error.to_string().contains("does not exist"),
        "the insert failed with {error}"
    );
    inserting.commit().unwrap();
    assert_eq!(orphans(&db), 0);
}

#[test]
fn a_snapshot_parent_delete_sees_a_child_committed_after_it_began() {
    for on_delete in ["RESTRICT", "CASCADE", "SET NULL"] {
        let db = open(
            &format!("fk_conc_snapshot_parent_{}", on_delete.replace(' ', "_")),
            on_delete,
        );
        let other = db.clone();
        let mut deleting = other
            .begin_with_isolation(IsolationLevel::SnapshotIsolation)
            .unwrap();
        assert_eq!(
            deleting
                .query_one::<i64, _>("SELECT COUNT(*) FROM c", ())
                .unwrap(),
            0
        );
        db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap();

        let error = deleting
            .execute("DELETE FROM p WHERE id = 1", ())
            .unwrap_err();
        assert_eq!(
            is_conflict(&error),
            on_delete != "RESTRICT",
            "{on_delete}: the delete failed with {error}"
        );
        deleting.commit().unwrap();
        assert_eq!(orphans(&db), 0, "{on_delete}");
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM c WHERE pid = 1", ())
                .unwrap(),
            1,
            "{on_delete}"
        );
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_child_insert_while_a_parent_delete_becomes_visible_is_a_write_conflict() {
    let db = open("fk_conc_child_while_visible", "RESTRICT");
    let inserting = db.clone();
    let insert = std::rc::Rc::new(std::cell::RefCell::new(None));
    let result = std::rc::Rc::clone(&insert);
    stoolap::test_failpoints::before_commit_visible(move || {
        *result.borrow_mut() = Some(inserting.execute("INSERT INTO c VALUES (1, 1)", ()));
    });
    db.execute("DELETE FROM p WHERE id = 1", ()).unwrap();

    let insert = insert.borrow_mut().take().unwrap();
    let error = insert.unwrap_err();
    assert!(is_conflict(&error), "the insert failed with {error}");
    assert_eq!(orphans(&db), 0);
}
