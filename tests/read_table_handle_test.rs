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

//! A read of a table its transaction holds no rows of and cannot write
//! while the statement runs registers no transaction-local store, and a
//! read the statement's own writes may reach still sees them

#![cfg(feature = "test-failpoints")]

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap()
}

#[test]
fn a_select_in_a_transaction_registers_no_store_until_a_write() {
    let db = Database::open("memory://read_table_handle_transaction").unwrap();
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO w VALUES (1, 10)", ()).unwrap();
    let engine = db.engine().clone();
    let before = engine.txn_store_entries();
    db.execute("BEGIN", ()).unwrap();
    let read = count(&db, "SELECT COUNT(*) FROM w WHERE x >= 0");
    let after_read = engine.txn_store_entries();
    db.execute("INSERT INTO w VALUES (2, 20)", ()).unwrap();
    let after_write = engine.txn_store_entries();
    let own = count(&db, "SELECT COUNT(*) FROM w WHERE x >= 0");
    db.execute("ROLLBACK", ()).unwrap();
    let rolled_back = count(&db, "SELECT COUNT(*) FROM w WHERE x >= 0");
    assert_eq!(
        (read, after_read, after_write, own, rolled_back),
        (1, before, before + 1, 2, 1),
        "(rows read, stores after the read, after the write, rows read after it, after the rollback)"
    );
}

#[test]
fn a_returning_block_reading_a_cascaded_table_sees_the_cascade() {
    let db = Database::open("memory://read_table_handle_cascade").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, p INTEGER REFERENCES p(id) ON DELETE CASCADE)",
        "INSERT INTO p VALUES (1), (2)",
        "INSERT INTO c VALUES (10, 1), (20, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let delete = db
        .prepare(
            "DELETE FROM p WHERE id = $1 RETURNING id, (SELECT COUNT(*) FROM c WHERE c.p = p.id)",
        )
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    let row = delete.query((1,)).unwrap().next().unwrap().unwrap();
    let got = (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap());
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(
        got,
        (1, 0),
        "the deleted parent and its children the block counts after the cascade"
    );
}
