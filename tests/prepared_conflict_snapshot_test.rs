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

//! Under SNAPSHOT isolation an INSERT ... SELECT and its ON CONFLICT
//! subqueries read the statement's one snapshot

#![cfg(feature = "test-failpoints")]

use stoolap::Database;

#[test]
fn an_insert_select_and_its_conflict_subquery_read_one_snapshot() {
    let db = Database::open("memory://prepared_conflict_snapshot_select").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO t VALUES (1, 0)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 10)",
        "SET isolation_level = 'SNAPSHOT'",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    stoolap::test_failpoints::after_transaction_begun(move || {
        other
            .execute("UPDATE u SET x = 20 WHERE id = 1", ())
            .unwrap();
    });
    db.execute(
        "INSERT INTO t SELECT id, x FROM u ON CONFLICT (id) DO UPDATE SET \
         b = excluded.b + (SELECT x FROM u WHERE u.id = excluded.id)",
        (),
    )
    .unwrap();
    let b: i64 = db.query_one("SELECT b FROM t WHERE id = 1", ()).unwrap();
    assert_eq!(
        b, 20,
        "the SELECT and the conflict subquery read different snapshots"
    );
}
