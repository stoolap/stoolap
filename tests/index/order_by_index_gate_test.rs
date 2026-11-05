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

//! ORDER BY one column + LIMIT through an index without an order sorts the
//! index's distinct values only while they are few against the rows; past
//! that the scan answers, the same rows either way

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::ordered_values_sorted_count;
use stoolap::Database;

fn column(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| format!("{:?}", r.unwrap().get_value(0)))
        .collect()
}

/// 3000 rows over `distinct` keys in `t` with the index and in `u` without
fn create(db: &Database, ty: &str, using: &str, distinct: i64) {
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, c {ty})"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, value * 7919 % {distinct} FROM generate_series(1, 3000)"
            ),
            (),
        )
        .unwrap();
    }
    db.execute(&format!("CREATE INDEX t_c ON t(c) USING {using}"), ())
        .unwrap();
}

#[test]
fn an_unordered_index_sorts_its_values_only_while_they_are_few() {
    for (ty, using, distinct, sorts) in [
        ("INTEGER", "HASH", 10, true),
        ("INTEGER", "HASH", 400, true),
        ("INTEGER", "HASH", 1000, false),
        ("INTEGER", "HASH", 3000, false),
        ("TEXT", "HASH", 10, true),
        ("TEXT", "HASH", 3000, false),
        ("INTEGER", "BITMAP", 10, true),
        ("INTEGER", "BITMAP", 3000, false),
    ] {
        let db = Database::open(&format!(
            "memory://order_by_index_gate_{ty}_{using}_{distinct}"
        ))
        .unwrap();
        create(&db, ty, using, distinct);
        for order in ["ASC", "DESC"] {
            let sql = format!("SELECT c FROM {{t}} ORDER BY c {order} LIMIT 5");
            let case = format!("{ty} {using} {distinct} distinct: {sql}");
            let before = ordered_values_sorted_count();
            let got = column(&db, &sql.replace("{t}", "t"));
            let sorted = ordered_values_sorted_count() - before;
            assert_eq!(got, column(&db, &sql.replace("{t}", "u")), "{case}");
            if sorts {
                assert_eq!(sorted, distinct as usize, "{case}: values sorted");
            } else {
                assert_eq!(sorted, 0, "{case}: values sorted");
            }
        }
    }
}
