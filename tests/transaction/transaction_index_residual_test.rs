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

//! Inside a transaction, a row inserted or updated by that transaction
//! must satisfy a secondary-index lookup together with a residual
//! predicate on the columns it just changed.

use stoolap::Database;

fn setup(dsn: &str) -> Database {
    let db = Database::open(dsn).expect("Failed to create database");
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 0)", ()).unwrap();
    db
}

fn count(tx: &mut stoolap::api::Transaction, sql: &str) -> i64 {
    tx.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

#[test]
fn test_same_tx_update_is_visible_through_index_with_residual() {
    let db = setup("memory://tx_index_residual_update");
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (2, 20, 0)", ()).unwrap();
    assert_eq!(
        tx.execute("UPDATE t SET v = 2 WHERE k = 20", ()).unwrap(),
        1
    );

    assert_eq!(count(&mut tx, "SELECT v FROM t WHERE k = 20"), 2);
    assert_eq!(
        count(&mut tx, "SELECT COUNT(*) FROM t WHERE k = 20 AND v = 2"),
        1
    );
    assert_eq!(
        count(&mut tx, "SELECT COUNT(*) FROM t WHERE k = 20 AND v = 0"),
        0
    );
    assert_eq!(
        tx.execute("UPDATE t SET v = 3 WHERE k = 20 AND v = 2", ())
            .unwrap(),
        1
    );
    assert_eq!(
        tx.execute("DELETE FROM t WHERE k = 20 AND v = 3", ())
            .unwrap(),
        1
    );
    assert_eq!(count(&mut tx, "SELECT COUNT(*) FROM t WHERE k = 20"), 0);
    tx.rollback().unwrap();
}

#[test]
fn test_same_tx_update_of_committed_row_is_visible_through_index_with_residual() {
    let db = setup("memory://tx_index_residual_committed");
    let mut tx = db.begin().unwrap();
    assert_eq!(
        tx.execute("UPDATE t SET v = 5 WHERE id = 1", ()).unwrap(),
        1
    );
    assert_eq!(
        count(&mut tx, "SELECT COUNT(*) FROM t WHERE k = 10 AND v = 5"),
        1
    );
    assert_eq!(
        tx.execute("DELETE FROM t WHERE k = 10 AND v = 5", ())
            .unwrap(),
        1
    );
    tx.rollback().unwrap();
    let outside: i64 = db
        .query("SELECT v FROM t WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(outside, 0);
}

/// A key changed inside the transaction is found under its new value and
/// no longer under the old one, even though the committed index still
/// carries the old value.
#[test]
fn test_same_tx_key_change_is_visible_through_index() {
    let db = setup("memory://tx_index_key_change");
    let mut tx = db.begin().unwrap();
    assert_eq!(
        tx.execute("UPDATE t SET k = 20 WHERE id = 1", ()).unwrap(),
        1
    );
    assert_eq!(
        count(&mut tx, "SELECT COUNT(*) FROM t WHERE k = 20 AND v = 0"),
        1
    );
    assert_eq!(
        count(&mut tx, "SELECT COUNT(*) FROM t WHERE k = 10 AND v = 0"),
        0
    );
    assert_eq!(
        tx.execute("UPDATE t SET v = 9 WHERE k = 20 AND v = 0", ())
            .unwrap(),
        1
    );
    assert_eq!(
        tx.execute("DELETE FROM t WHERE k = 20 AND v = 9", ())
            .unwrap(),
        1
    );
    assert_eq!(count(&mut tx, "SELECT COUNT(*) FROM t"), 0);
    tx.rollback().unwrap();
    let outside: i64 = db
        .query("SELECT COUNT(*) FROM t WHERE k = 10", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(outside, 1);
}
