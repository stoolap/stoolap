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

//! A member test of an uncorrelated block asks the block once per
//! statement: every row after the first tests the members the run kept,
//! and the answers stay those of the same test over a literal list

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::kept_member_calls;
use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort();
    out
}

#[test]
fn a_member_test_asks_its_block_once_per_statement() {
    let db = Database::open("memory://prepared_kept_members").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, i INTEGER)",
        "INSERT INTO t SELECT g.value, CASE WHEN g.value % 10 = 0 THEN NULL ELSE g.value % 7 END \
         FROM generate_series(1, 50) g",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO m VALUES (1, 1, 3), (2, 1, 5), (3, 2, 4), (4, 2, NULL)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (test, same) in [
        ("i + 0 IN (SELECT v FROM m WHERE k = 1)", "i + 0 IN (3, 5)"),
        (
            "i + 0 NOT IN (SELECT v FROM m WHERE k = 1)",
            "i + 0 NOT IN (3, 5)",
        ),
        (
            "i + 0 IN (SELECT v FROM m WHERE k = 2)",
            "i + 0 IN (4, NULL)",
        ),
        ("i + 0 > ANY (SELECT v FROM m WHERE k = 1)", "i + 0 > 3"),
        ("i + 0 >= ALL (SELECT v FROM m WHERE k = 1)", "i + 0 >= 5"),
        (
            "i + 0 <> ANY (SELECT v FROM m WHERE k = 1)",
            "i IS NOT NULL",
        ),
        (
            "(i + 0 IN (SELECT v FROM m WHERE k = 1) OR id = 7)",
            "(i + 0 IN (3, 5) OR id = 7)",
        ),
    ] {
        let sql = format!("SELECT id FROM t WHERE {test}");
        let before = kept_member_calls();
        let got = ids(&db, &sql);
        assert_eq!(
            kept_member_calls() - before,
            1,
            "{sql}: the block asked once"
        );
        assert_eq!(
            got,
            ids(&db, &format!("SELECT id FROM t WHERE {same}")),
            "{sql}"
        );
        assert!(!got.is_empty(), "{sql}");
    }
}
