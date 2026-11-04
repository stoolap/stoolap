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

//! The pre-compiled memory-WHERE path must carry the execution context's
//! transaction id, matching the evaluator path it replaced.

use stoolap::api::Database;

#[test]
fn update_memory_where_sees_transaction_id() {
    let db = Database::open("memory://dmlctx_upd").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();

    let mut tx = db.begin().unwrap();
    // CURRENT_TRANSACTION_ID() is not pushable, forcing the memory filter
    let n = tx
        .execute(
            "UPDATE t SET a = 1 WHERE id + 0 = 1 AND CURRENT_TRANSACTION_ID() > 0",
            (),
        )
        .unwrap();
    assert_eq!(n, 1, "WHERE must see the transaction id, not NULL");
    tx.commit().unwrap();
    let a: i64 = db.query_one("SELECT a FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(a, 1);
}

#[test]
fn delete_memory_where_sees_transaction_id() {
    let db = Database::open("memory://dmlctx_del").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 0)", ()).unwrap();

    let mut tx = db.begin().unwrap();
    let n = tx
        .execute(
            "DELETE FROM t WHERE id + 0 = 1 AND CURRENT_TRANSACTION_ID() > 0",
            (),
        )
        .unwrap();
    assert_eq!(n, 1);
    tx.commit().unwrap();
    let c: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(c, 0);
}
