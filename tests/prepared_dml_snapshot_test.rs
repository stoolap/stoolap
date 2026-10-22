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

//! A prepared UPDATE or DELETE reads its target and its subqueries at one
//! point, also when another handle commits once its transaction has begun

#![cfg(feature = "test-failpoints")]

use stoolap::Database;

fn table(db: &Database) -> String {
    db.query("SELECT id, b FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            format!("{}:{}", r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn a_prepared_update_or_delete_reads_one_point() {
    let mut wrong = Vec::new();
    for level in ["READ COMMITTED", "SNAPSHOT"] {
        for (sql, left) in [
            (
                "UPDATE t SET b = (SELECT MAX(b) FROM t) WHERE id = 1",
                "1:40 2:20 3:30 4:40 9:90",
            ),
            (
                "DELETE FROM t WHERE b < (SELECT MAX(b) FROM t)",
                "4:40 9:90",
            ),
        ] {
            let db = Database::open(&format!("memory://prepared_dml_snapshot_{level}")).unwrap();
            for setup in [
                "CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)",
                "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40)",
            ] {
                db.execute(setup, ()).unwrap();
            }
            db.execute(&format!("SET isolation_level = '{level}'"), ())
                .unwrap();
            let other = db.clone();
            stoolap::test_failpoints::after_transaction_begun(move || {
                other.execute("INSERT INTO t VALUES (9, 90)", ()).unwrap();
            });
            db.execute(sql, ()).unwrap();
            let got = table(&db);
            if got != left {
                wrong.push(format!("{sql} ({level}): {got}, want {left}"));
            }
            db.execute("DROP TABLE t", ()).unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
