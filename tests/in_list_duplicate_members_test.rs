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
        for table in ["btree", "hash"] {
            assert_eq!(
                ids(db, &sql.replace("{t}", table)),
                want,
                "{phase}, {table}: k IN ({list})"
            );
        }
    }
}

fn create(db: &Database) {
    for (table, index) in [("plain", ""), ("btree", "BTREE"), ("hash", "HASH")] {
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
fn a_row_moved_between_two_member_probes_returns_once() {
    let db = Database::open("memory://a_row_moved_between_two_member_probes").unwrap();
    let other = db.clone();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_k ON t(k) USING BTREE", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (6, 3), (7, 1), (8, 2)", ())
        .unwrap();
    let moved = std::rc::Rc::new(std::cell::Cell::new(false));
    let moved_in_hook = std::rc::Rc::clone(&moved);
    stoolap::test_failpoints::after_in_member_probed(move || {
        other
            .execute("UPDATE t SET k = 2 WHERE id = 7", ())
            .unwrap();
        moved_in_hook.set(true);
    });
    assert_eq!(ids(&db, "SELECT id FROM t WHERE k IN (1, 2)"), vec![7, 8]);
    assert!(moved.get(), "the row moved between the two probes");
}
