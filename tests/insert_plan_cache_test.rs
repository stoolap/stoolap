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

//! CHECK constraints and ON DUPLICATE KEY UPDATE clauses compiled once per
//! statement or cached with its plan still enforce and apply what the table
//! holds now

use stoolap::{Database, Error};

fn is_check_violation(result: stoolap::Result<i64>) -> bool {
    matches!(result, Err(Error::CheckConstraintViolation { .. }))
}

fn value(db: &Database, sql: &str) -> i64 {
    db.query_one::<i64, _>(sql, ()).unwrap()
}

#[test]
fn test_transaction_insert_enforces_check() {
    let db = Database::open("memory://insert_plan_cache_tx_check").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v >= 0))",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (1, 5), (2, 6)", ())
        .unwrap();
    assert!(
        is_check_violation(tx.execute("INSERT INTO t VALUES (3, 7), (4, -1)", ())),
        "the second row breaks the CHECK"
    );
    tx.commit().unwrap();
    assert_eq!(value(&db, "SELECT COUNT(*) FROM t"), 2);
}

#[test]
fn test_upsert_enforces_check_on_insert_and_update() {
    let db = Database::open("memory://insert_plan_cache_upsert_check").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v >= 0))",
        (),
    )
    .unwrap();
    let upsert = "INSERT INTO t VALUES ($1, $2) ON DUPLICATE KEY UPDATE v = $2";
    db.execute(upsert, (1, 5)).unwrap();
    db.execute(upsert, (1, 6)).unwrap();
    assert!(is_check_violation(db.execute(upsert, (2, -1))), "insert");
    assert!(is_check_violation(db.execute(upsert, (1, -1))), "update");
    assert_eq!(value(&db, "SELECT COUNT(*) FROM t"), 1);
    assert_eq!(value(&db, "SELECT v FROM t WHERE id = 1"), 6);
}

#[test]
fn test_cached_upsert_follows_a_column_dropped_through_another_handle() {
    let db = Database::open("memory://insert_plan_cache_drop_column").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, v INTEGER CHECK (v < 100))",
        (),
    )
    .unwrap();
    let upsert = "INSERT INTO t (id, v) VALUES ($1, $2) ON DUPLICATE KEY UPDATE v = $2";
    db.execute(upsert, (1, 1)).unwrap();
    db.execute(upsert, (1, 2)).unwrap();
    db.clone()
        .execute("ALTER TABLE t DROP COLUMN a", ())
        .unwrap();
    db.execute(upsert, (1, 3)).unwrap();
    assert_eq!(value(&db, "SELECT v FROM t WHERE id = 1"), 3);
    assert!(
        is_check_violation(db.execute(upsert, (1, 200))),
        "the CHECK on v still applies after the columns moved"
    );
    assert_eq!(value(&db, "SELECT v FROM t WHERE id = 1"), 3);
}

#[test]
fn test_repeated_do_nothing_keeps_the_first_row() {
    let db = Database::open("memory://insert_plan_cache_do_nothing").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER CHECK (v >= 0))",
        (),
    )
    .unwrap();
    let insert = "INSERT INTO t VALUES ($1, $2) ON CONFLICT (id) DO NOTHING";
    for (id, v) in [(1, 1), (1, 2), (2, 3), (2, 4)] {
        db.execute(insert, (id, v)).unwrap();
    }
    assert!(is_check_violation(db.execute(insert, (3, -1))));
    assert_eq!(value(&db, "SELECT SUM(v) FROM t"), 4);
}
