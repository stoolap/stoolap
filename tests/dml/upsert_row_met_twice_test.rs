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

//! An upsert applies its rows in order, so a later row may update a row an
//! earlier one of the same statement inserted or updated

use stoolap::Database;

fn rows(db: &Database) -> Vec<(i64, i64)> {
    db.query("SELECT id, n FROM t ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect()
}

#[test]
fn a_later_row_updates_the_row_an_earlier_one_wrote() {
    let cases: [(&str, [(i64, i64); 2]); 4] = [
        (
            "INSERT INTO t VALUES (1, 5), (1, 6) ON CONFLICT (id) DO UPDATE SET n = excluded.n",
            [(1, 6), (2, 20)],
        ),
        (
            "INSERT INTO t VALUES (1, 5), (1, 6) ON DUPLICATE KEY UPDATE n = n + 1",
            [(1, 12), (2, 20)],
        ),
        (
            "INSERT INTO t VALUES (3, 1), (3, 1), (3, 1) ON CONFLICT (id) DO UPDATE SET n = n + 1",
            [(1, 10), (2, 20)],
        ),
        (
            "INSERT INTO t VALUES (3, 1), (3, 1), (3, 1) ON DUPLICATE KEY UPDATE n = n + 1",
            [(1, 10), (2, 20)],
        ),
    ];
    for (at, (sql, kept)) in cases.into_iter().enumerate() {
        let db = Database::open(&format!("memory://upsert_row_met_twice_{at}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        db.execute(sql, ()).unwrap();
        let mut expected = kept.to_vec();
        if sql.contains("(3, 1)") {
            expected.push((3, 3));
        }
        assert_eq!(rows(&db), expected, "{sql}");
    }
}
