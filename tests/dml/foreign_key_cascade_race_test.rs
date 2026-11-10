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

#![cfg(feature = "test-failpoints")]

use stoolap::{test_failpoints, Database};

#[test]
fn a_cascade_leaves_a_child_that_moved_to_another_parent_after_it_was_read() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://fk_cascade_race_moved_child").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON UPDATE CASCADE)",
        "INSERT INTO p VALUES (1), (3)",
        "INSERT INTO c VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let writer = db.clone();
    test_failpoints::after_cascade_targets_read(move || {
        std::thread::spawn(move || writer.execute("UPDATE c SET pid = 3 WHERE id = 1", ()))
            .join()
            .unwrap()
            .unwrap();
    });
    db.execute("UPDATE p SET id = 2 WHERE id = 1", ()).unwrap();
    let pid: i64 = db.query_one("SELECT pid FROM c WHERE id = 1", ()).unwrap();
    assert_eq!(pid, 3, "the cascade rewrote a child that no longer named 1");
}

#[test]
fn a_cascade_refuses_a_child_whose_layout_changed_after_it_was_read() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://fk_cascade_race_dropped_column").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pad INTEGER, \
         pid INTEGER REFERENCES p(id) ON UPDATE CASCADE, x INTEGER)",
        "INSERT INTO p VALUES (1)",
        "INSERT INTO c VALUES (1, 0, 1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    test_failpoints::after_cascade_targets_read(move || {
        std::thread::spawn(move || other.execute("ALTER TABLE c DROP COLUMN pad", ()))
            .join()
            .unwrap()
            .unwrap();
    });
    let error = db
        .execute("UPDATE p SET id = 2 WHERE id = 1", ())
        .unwrap_err()
        .to_string();
    assert!(error.to_lowercase().contains("schema"), "{error}");
    let row: (i64, i64) = {
        let r = db
            .query("SELECT pid, x FROM c", ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        (r.get(0).unwrap(), r.get(1).unwrap())
    };
    assert_eq!(row, (1, 1));
    let id: i64 = db.query_one("SELECT id FROM p", ()).unwrap();
    assert_eq!(id, 1);
}

#[test]
fn a_delete_cascade_refuses_when_a_child_it_read_moved_to_another_parent() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://fk_cascade_race_delete_moved_child").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE CASCADE)",
        "CREATE TABLE g (id INTEGER PRIMARY KEY, cid INTEGER REFERENCES c(id) ON DELETE CASCADE)",
        "INSERT INTO p VALUES (1), (3)",
        "INSERT INTO c VALUES (1, 1)",
        "INSERT INTO g VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let writer = db.clone();
    test_failpoints::after_cascade_targets_read(move || {
        std::thread::spawn(move || writer.execute("UPDATE c SET pid = 3 WHERE id = 1", ()))
            .join()
            .unwrap()
            .unwrap();
    });
    let error = db
        .execute("DELETE FROM p WHERE id = 1", ())
        .unwrap_err()
        .to_string();
    assert!(error.contains("write conflict"), "{error}");
    for (sql, want) in [
        ("SELECT COUNT(*) FROM p", 2),
        ("SELECT pid FROM c WHERE id = 1", 3),
        ("SELECT COUNT(*) FROM g", 1),
    ] {
        assert_eq!(db.query_one::<i64, _>(sql, ()).unwrap(), want, "{sql}");
    }
}

#[test]
fn a_delete_set_null_leaves_a_child_that_moved_to_another_parent_after_it_was_read() {
    let _guard = test_failpoints::FailpointGuard::new();
    let db = Database::open("memory://fk_cascade_race_delete_null_moved").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER REFERENCES p(id) ON DELETE SET NULL)",
        "INSERT INTO p VALUES (1), (3)",
        "INSERT INTO c VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let writer = db.clone();
    test_failpoints::after_cascade_targets_read(move || {
        std::thread::spawn(move || writer.execute("UPDATE c SET pid = 3 WHERE id = 1", ()))
            .join()
            .unwrap()
            .unwrap();
    });
    db.execute("DELETE FROM p WHERE id = 1", ()).unwrap();
    let pid: i64 = db.query_one("SELECT pid FROM c WHERE id = 1", ()).unwrap();
    assert_eq!(
        pid, 3,
        "the SET NULL cleared a child that no longer named 1"
    );
}
