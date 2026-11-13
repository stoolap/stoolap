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
fn test_alter_auto_increment_lowers_to_past_the_largest_row() {
    let db = open("alter_ai_lower");
    db.execute("INSERT INTO t (v) VALUES ('a'), ('b'), ('c')", ())
        .unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 2", ()).unwrap();
    assert_eq!(insert(&db, "d"), 4, "one past the largest row");
    db.execute("DELETE FROM t WHERE id = 4", ()).unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 2", ()).unwrap();
    assert_eq!(insert(&db, "e"), 4, "a deleted row's id is taken again");
    db.execute("INSERT INTO t (id, v) VALUES (500, 'x')", ())
        .unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 100", ())
        .unwrap();
    assert_eq!(insert(&db, "f"), 501, "an explicit id counts");
    db.execute("DELETE FROM t WHERE id >= 500", ()).unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 100", ())
        .unwrap();
    assert_eq!(insert(&db, "g"), 100);
}

#[test]
fn test_alter_auto_increment_below_a_rolled_back_id() {
    let db = open("alter_ai_rolled_back");
    assert_eq!(insert(&db, "a"), 1);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t (id, v) VALUES (100, 'b')", ())
        .unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 50", ()).unwrap();
    assert_eq!(insert(&db, "c"), 50);
}

#[test]
fn test_alter_auto_increment_lowering_waits_for_uncommitted_rows() {
    let db = open("alter_ai_pending");
    let other = db.clone();
    db.execute("INSERT INTO t (id, v) VALUES (100, 'a')", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 100", ()).unwrap();
    for row in [
        "INSERT INTO t (v) VALUES ('b')",
        "INSERT INTO t (id, v) VALUES (70, 'b')",
    ] {
        other.execute("BEGIN", ()).unwrap();
        other.execute(row, ()).unwrap();
        let message = error(&db, "ALTER TABLE t AUTO_INCREMENT = 50");
        assert!(message.contains("cannot be lowered"), "{row}: {message}");
        other.execute("ROLLBACK", ()).unwrap();
    }
    db.execute("ALTER TABLE t AUTO_INCREMENT = 50", ()).unwrap();
    assert_eq!(insert(&db, "c"), 50, "rolled back rows hold no id");
}

#[test]
fn test_alter_auto_increment_raises_beside_uncommitted_rows() {
    let db = open("alter_ai_pending_raise");
    let other = db.clone();
    other.execute("BEGIN", ()).unwrap();
    other.execute("INSERT INTO t (v) VALUES ('a')", ()).unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 1000", ())
        .unwrap();
    other.execute("COMMIT", ()).unwrap();
    assert_eq!(insert(&db, "b"), 1000);
}

#[test]
fn test_alter_auto_increment_lowering_in_a_transaction_that_inserted() {
    let db = open("alter_ai_own_rows");
    db.execute("INSERT INTO t (id, v) VALUES (100, 'a')", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 100", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    assert_eq!(insert(&db, "b"), 101);
    let message = error(&db, "ALTER TABLE t AUTO_INCREMENT = 50");
    assert!(message.contains("cannot be lowered"), "{message}");
    db.execute("COMMIT", ()).unwrap();
    db.execute("ALTER TABLE t AUTO_INCREMENT = 50", ()).unwrap();
    assert_eq!(insert(&db, "c"), 102, "the committed row 101 counts");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_alter_auto_increment_insert_during_lowering_takes_the_lowered_id() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let db = open("alter_ai_insert_during");
    db.execute("INSERT INTO t (id, v) VALUES (100, 'a')", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 100", ()).unwrap();
    let other = db.clone();
    let started = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&started);
    stoolap::test_failpoints::on_inserters_closed(move || {
        let (sent, seen) = std::sync::mpsc::channel();
        let waiting = sent.clone();
        let thread = std::thread::spawn(move || {
            stoolap::test_failpoints::on_inserter_waiting(move || {
                let _ = waiting.send(None);
            });
            let id = insert(&other, "b");
            let _ = sent.send(Some(id));
            id
        });
        *slot.borrow_mut() = Some((thread, seen.recv().unwrap()));
    });
    db.execute("ALTER TABLE t AUTO_INCREMENT = 50", ()).unwrap();
    let (thread, first) = started.take().expect("the lowering ran its hook");
    assert_eq!(first, None, "the insert waited for the lowering");
    assert_eq!(thread.join().unwrap(), 50);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_alter_auto_increment_lowering_during_a_commit_is_refused() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let db = open("alter_ai_during_commit");
    db.execute("INSERT INTO t (id, v) VALUES (100, 'a')", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 100", ()).unwrap();
    let other = db.clone();
    let outcome = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&outcome);
    stoolap::test_failpoints::after_indexes_published(move || {
        let result = other.execute("ALTER TABLE t AUTO_INCREMENT = 50", ());
        *slot.borrow_mut() = Some(result.map_or_else(|e| e.to_string(), |_| "lowered".into()));
    });
    db.execute("INSERT INTO t (id, v) VALUES (70, 'b')", ())
        .unwrap();
    let outcome = outcome.take().expect("the commit ran its hook");
    assert!(outcome.contains("cannot be lowered"), "{outcome}");
    db.execute("ALTER TABLE t AUTO_INCREMENT = 50", ()).unwrap();
    assert_eq!(insert(&db, "c"), 71, "the committed row 70 counts");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_alter_auto_increment_lowering_keeps_a_row_whose_deletion_commits() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let db = open("alter_ai_during_delete");
    db.execute("INSERT INTO t (id, v) VALUES (100, 'a')", ())
        .unwrap();
    let other = db.clone();
    let lowered = Rc::new(RefCell::new(None));
    let slot = Rc::clone(&lowered);
    stoolap::test_failpoints::before_commit_visible(move || {
        let result = other.execute("ALTER TABLE t AUTO_INCREMENT = 50", ());
        *slot.borrow_mut() = Some(result.map(|_| ()).map_err(|e| e.to_string()));
    });
    db.execute("DELETE FROM t WHERE id = 100", ()).unwrap();
    let lowered = lowered.take().expect("the commit ran its hook");
    assert_eq!(lowered, Ok(()));
    assert_eq!(
        insert(&db, "b"),
        101,
        "a deletion not yet visible may still be taken back"
    );
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
