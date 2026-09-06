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

//! Savepoints form a stack: RELEASE drops the named savepoint and every
//! later one, ROLLBACK TO drops the later ones and keeps its own, and a
//! name may be reused. The same statements work through the Transaction API.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db
}

fn run(db: &Database, statements: &[&str]) {
    for sql in statements {
        db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
}

fn count(db: &Database) -> i64 {
    db.query("SELECT COUNT(*) FROM t", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .next()
        .unwrap()
}

#[test]
fn test_release_drops_the_savepoints_opened_after_it() {
    let db = setup("savepoint_release_drops_later");
    run(
        &db,
        &[
            "BEGIN",
            "SAVEPOINT a",
            "INSERT INTO t VALUES (1)",
            "SAVEPOINT b",
            "INSERT INTO t VALUES (2)",
            "RELEASE SAVEPOINT a",
        ],
    );
    assert!(db.execute("ROLLBACK TO SAVEPOINT b", ()).is_err());
    assert!(db.execute("RELEASE SAVEPOINT a", ()).is_err());
    assert_eq!(count(&db), 2);
    run(&db, &["COMMIT"]);
    assert_eq!(count(&db), 2);
}

#[test]
fn test_release_without_the_savepoint_keyword() {
    let db = setup("savepoint_release_bare");
    run(
        &db,
        &[
            "BEGIN",
            "SAVEPOINT a",
            "INSERT INTO t VALUES (1)",
            "RELEASE a",
            "COMMIT",
        ],
    );
    assert_eq!(count(&db), 1);
}

#[test]
fn test_rollback_to_keeps_its_savepoint_and_drops_later_ones() {
    let db = setup("savepoint_rollback_keeps_own");
    run(
        &db,
        &[
            "BEGIN",
            "SAVEPOINT a",
            "SAVEPOINT b",
            "INSERT INTO t VALUES (1)",
            "ROLLBACK TO SAVEPOINT a",
        ],
    );
    assert!(db.execute("RELEASE SAVEPOINT b", ()).is_err());
    run(
        &db,
        &["INSERT INTO t VALUES (2)", "ROLLBACK TO SAVEPOINT a"],
    );
    assert_eq!(count(&db), 0);
    run(&db, &["RELEASE SAVEPOINT a", "COMMIT"]);
    assert_eq!(count(&db), 0);
}

#[test]
fn test_a_reused_name_stacks() {
    let db = setup("savepoint_reused_name");
    run(
        &db,
        &[
            "BEGIN",
            "SAVEPOINT a",
            "INSERT INTO t VALUES (1)",
            "SAVEPOINT a",
            "INSERT INTO t VALUES (2)",
            "ROLLBACK TO SAVEPOINT a",
        ],
    );
    assert_eq!(count(&db), 1);
    run(&db, &["RELEASE SAVEPOINT a", "ROLLBACK TO SAVEPOINT a"]);
    assert_eq!(count(&db), 0);
    run(&db, &["RELEASE SAVEPOINT a"]);
    assert!(db.execute("RELEASE SAVEPOINT a", ()).is_err());
    run(&db, &["ROLLBACK"]);
}

#[test]
fn test_savepoints_through_the_transaction_api() {
    let db = setup("savepoint_transaction_api");
    let mut tx = db.begin().unwrap();
    tx.execute("SAVEPOINT a", ()).unwrap();
    tx.execute("INSERT INTO t VALUES (1)", ()).unwrap();
    tx.execute("SAVEPOINT b", ()).unwrap();
    tx.execute("INSERT INTO t VALUES (2)", ()).unwrap();
    tx.execute("ROLLBACK TO SAVEPOINT b", ()).unwrap();
    tx.execute("RELEASE SAVEPOINT a", ()).unwrap();
    assert!(tx.execute("RELEASE SAVEPOINT b", ()).is_err());
    tx.commit().unwrap();
    assert_eq!(count(&db), 1);
}

#[test]
fn test_release_stays_usable_as_a_column_name() {
    let db = Database::open("memory://savepoint_release_column").unwrap();
    db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY, release TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO r VALUES (1, 'v1')", ()).unwrap();
    let release: Vec<String> = db
        .query("SELECT release FROM r WHERE release = 'v1'", ())
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(release, ["v1"]);
    run(
        &db,
        &[
            "ALTER TABLE r RENAME COLUMN release TO version",
            "ALTER TABLE r ADD COLUMN release TEXT",
            "ALTER TABLE r DROP COLUMN release",
        ],
    );
}
