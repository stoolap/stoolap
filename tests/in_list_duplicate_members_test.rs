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

//! An IN list naming one value more than once, directly or through members
//! the column reads as equal, returns each matching row once, with or
//! without an index on the column

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn check(db: &Database, phase: &str) {
    for list in ["1, 1", "1, '1'", "1, 1.0", "2, 1, 2, 1", "1, 2, 1, 3"] {
        let sql = format!("SELECT id FROM {{t}} WHERE k IN ({list})");
        let want = ids(db, &sql.replace("{t}", "plain"));
        for table in ["btree", "hash", "bitmap"] {
            assert_eq!(
                ids(db, &sql.replace("{t}", table)),
                want,
                "{phase}, {table}: k IN ({list})"
            );
        }
    }
}

fn create(db: &Database) {
    for (table, index) in [
        ("plain", ""),
        ("btree", "BTREE"),
        ("hash", "HASH"),
        ("bitmap", "BITMAP"),
    ] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER NOT NULL)"),
            (),
        )
        .unwrap();
        if !index.is_empty() {
            db.execute(
                &format!("CREATE INDEX {table}_k ON {table}(k) USING {index}"),
                (),
            )
            .unwrap();
        }
        db.execute(
            &format!("INSERT INTO {table} SELECT value, value % 5 FROM generate_series(1, 50)"),
            (),
        )
        .unwrap();
    }
}

#[test]
fn duplicate_members_return_each_row_once_in_memory() {
    let db = Database::open("memory://duplicate_members_return_each_row_once").unwrap();
    create(&db);
    check(&db, "memory");
}

#[test]
fn duplicate_members_return_each_row_once_hot_and_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    create(&db);
    check(&db, "hot");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed");
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_row_cannot_move_between_two_member_probes() {
    let db = Database::open("memory://a_row_cannot_move_between_two_member_probes").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, s TEXT)",
        "CREATE INDEX t_k ON t(k) USING BTREE",
        "INSERT INTO t VALUES (6, 3, 'a'), (7, 1, 'b'), (8, 2, 'c')",
        "CREATE TABLE p (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO p VALUES (1, 1), (2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut failed = Vec::new();
    for sql in [
        "SELECT id FROM t WHERE k IN (1, 2)",
        "SELECT id FROM t WHERE k IN (1, 2) AND s || 'x' <> 'zx'",
        "SELECT id FROM t WHERE k IN (SELECT a FROM p)",
        "SELECT id FROM t WHERE k IN (SELECT a FROM p) AND s || 'x' <> 'zx'",
        "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM p WHERE p.a = t.k) AND s || 'x' <> 'zx'",
    ] {
        let other = db.clone();
        let (at_index, reached) = std::sync::mpsc::channel();
        let (key_removed, removed) = std::sync::mpsc::channel();
        let mover = std::rc::Rc::new(std::cell::RefCell::new(None));
        let mover_in_hook = std::rc::Rc::clone(&mover);
        let moved_during_probe = std::rc::Rc::new(std::cell::Cell::new(false));
        let moved_in_hook = std::rc::Rc::clone(&moved_during_probe);
        stoolap::test_failpoints::after_in_member_probed(move || {
            *mover_in_hook.borrow_mut() = Some(std::thread::spawn(move || {
                stoolap::test_failpoints::before_commit_index_capture(move || {
                    let _ = at_index.send(());
                });
                stoolap::test_failpoints::after_commit_index_removed(move || {
                    let _ = key_removed.send(());
                });
                other
                    .execute("UPDATE t SET k = 2 WHERE id = 7", ())
                    .unwrap();
            }));
            reached
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("the UPDATE reached its index changes");
            let waited = removed.recv_timeout(std::time::Duration::from_millis(500));
            moved_in_hook.set(waited.is_ok());
        });
        assert_eq!(ids(&db, sql), vec![7, 8], "{sql}");
        let Some(mover) = mover.borrow_mut().take() else {
            stoolap::test_failpoints::after_in_member_probed(|| {});
            failed.push(format!("{sql}: the index did not read the members"));
            continue;
        };
        mover.join().unwrap();
        if moved_during_probe.get() {
            failed.push(format!("{sql}: the row moved while the members were read"));
        }
        assert_eq!(ids(&db, "SELECT id FROM t WHERE k = 2"), vec![7, 8]);
        db.execute("UPDATE t SET k = 1 WHERE id = 7", ()).unwrap();
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
fn count_member_probes(count: std::rc::Rc<std::cell::Cell<usize>>) {
    stoolap::test_failpoints::after_in_member_probed(move || {
        count.set(count.get() + 1);
        count_member_probes(count);
    });
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_limit_stops_the_member_read_at_the_key_that_meets_it() {
    let db = Database::open("memory://a_limit_stops_the_member_read_at_the_key").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL)",
        "CREATE INDEX t_k ON t(k) USING BTREE",
        "INSERT INTO t SELECT value, value FROM generate_series(1, 10)",
        "CREATE TABLE p (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO p SELECT value, value FROM generate_series(1, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let probes = std::rc::Rc::new(std::cell::Cell::new(0));
    count_member_probes(std::rc::Rc::clone(&probes));
    let rows = ids(&db, "SELECT id FROM t WHERE k IN (SELECT a FROM p) LIMIT 2");
    stoolap::test_failpoints::after_in_member_probed(|| {});
    assert_eq!(rows.len(), 2);
    assert_eq!(
        probes.get(),
        2,
        "each key holds one row, so two keys meet LIMIT 2 whatever their order"
    );
}
