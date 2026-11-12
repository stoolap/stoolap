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

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY AUTO_INCREMENT, v TEXT)",
        (),
    )
    .unwrap();
    db
}

fn insert(db: &Database, v: &str) -> i64 {
    db.execute(&format!("INSERT INTO t (v) VALUES ('{v}')"), ())
        .unwrap();
    db.query_one(&format!("SELECT id FROM t WHERE v = '{v}'"), ())
        .unwrap()
}

fn error(db: &Database, sql: &str) -> String {
    match db.execute(sql, ()) {
        Ok(_) => panic!("{sql} succeeded"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn test_alter_auto_increment_sets_the_next_id() {
    let db = open("alter_ai_next");
    db.execute("ALTER TABLE t AUTO_INCREMENT = 100", ())
        .unwrap();
    assert_eq!(insert(&db, "a"), 100);
    assert_eq!(insert(&db, "b"), 101);
}

#[test]
fn test_alter_auto_increment_spellings() {
    let db = open("alter_ai_spellings");
    db.execute("ALTER TABLE t AUTO_INCREMENT 10", ()).unwrap();
    assert_eq!(insert(&db, "a"), 10);
    db.execute("alter table t autoincrement = 20", ()).unwrap();
    assert_eq!(insert(&db, "b"), 20);
}

#[test]
fn test_alter_auto_increment_past_existing_rows() {
    let db = open("alter_ai_rows");
    db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
        .unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 50", ()).unwrap();
    assert_eq!(insert(&db, "d"), 50);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 4);
}

#[test]
fn test_alter_auto_increment_to_the_next_id_changes_nothing() {
    let db = open("alter_ai_same");
    db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
        .unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 4", ()).unwrap();
    assert_eq!(insert(&db, "d"), 4);
}

#[test]
fn test_alter_auto_increment_below_a_handed_out_id_is_refused() {
    let db = open("alter_ai_below");
    db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
        .unwrap();
    let message = error(&db, "ALTER TABLE t AUTO_INCREMENT = 3");
    assert!(
        message.contains("cannot go below 4"),
        "unexpected error: {message}"
    );
    db.execute("DELETE FROM t WHERE id = 3", ()).unwrap();
    let message = error(&db, "ALTER TABLE t AUTO_INCREMENT = 3");
    assert!(
        message.contains("cannot go below 4"),
        "a deleted id stays handed out: {message}"
    );
    db.execute("INSERT INTO t (id, v) VALUES (500, 'x')", ())
        .unwrap();
    let message = error(&db, "ALTER TABLE t AUTO_INCREMENT = 100");
    assert!(
        message.contains("cannot go below 501"),
        "an explicit id counts: {message}"
    );
    assert_eq!(insert(&db, "d"), 501, "a refused ALTER leaves the counter");
}

#[test]
fn test_alter_auto_increment_is_not_undone_by_a_rollback() {
    let db = open("alter_ai_rollback");
    db.execute("BEGIN", ()).unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 100", ())
        .unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(insert(&db, "a"), 100);
}

#[test]
fn test_alter_auto_increment_needs_an_auto_increment_column() {
    let db = Database::open("memory://alter_ai_no_column").unwrap();
    db.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE n (v TEXT)", ()).unwrap();
    for table in ["p", "n"] {
        let message = error(&db, &format!("ALTER TABLE {table} AUTO_INCREMENT = 10"));
        assert!(
            message.contains("has no AUTO_INCREMENT column"),
            "unexpected error for {table}: {message}"
        );
    }
    let message = error(&db, "ALTER TABLE missing AUTO_INCREMENT = 10");
    assert!(message.contains("missing"), "unexpected error: {message}");
}

#[test]
fn test_alter_auto_increment_value_must_be_a_positive_integer() {
    let db = open("alter_ai_values");
    for sql in [
        "ALTER TABLE t AUTO_INCREMENT = 0",
        "ALTER TABLE t AUTO_INCREMENT = -5",
        "ALTER TABLE t AUTO_INCREMENT = 'x'",
        "ALTER TABLE t AUTO_INCREMENT = 1.5",
        "ALTER TABLE t AUTO_INCREMENT =",
        "ALTER TABLE t AUTO_INCREMENT = 99999999999999999999",
    ] {
        error(&db, sql);
    }
    assert_eq!(insert(&db, "a"), 1, "a refused ALTER leaves the counter");
}
