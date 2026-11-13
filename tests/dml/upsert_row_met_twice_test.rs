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

//! ON DUPLICATE KEY UPDATE applies its rows in order, so a later row may
//! update a row an earlier one of the same statement inserted or updated;
//! ON CONFLICT DO UPDATE refuses such a row and writes nothing

use stoolap::Database;

type Case = (&'static str, Option<&'static [(i64, i64)]>);

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
fn a_later_row_meets_the_row_an_earlier_one_wrote() {
    let cases: [Case; 4] = [
        (
            "INSERT INTO t VALUES (1, 5), (1, 6) ON CONFLICT (id) DO UPDATE SET n = excluded.n",
            None,
        ),
        (
            "INSERT INTO t VALUES (1, 5), (1, 6) ON DUPLICATE KEY UPDATE n = n + 1",
            Some(&[(1, 12), (2, 20)]),
        ),
        (
            "INSERT INTO t VALUES (3, 1), (3, 1), (3, 1) ON CONFLICT (id) DO UPDATE SET n = n + 1",
            None,
        ),
        (
            "INSERT INTO t VALUES (3, 1), (3, 1), (3, 1) ON DUPLICATE KEY UPDATE n = n + 1",
            Some(&[(1, 10), (2, 20), (3, 3)]),
        ),
    ];
    for (at, (sql, kept)) in cases.into_iter().enumerate() {
        let db = Database::open(&format!("memory://upsert_row_met_twice_{at}")).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20)", ())
            .unwrap();
        let ran = db.execute(sql, ());
        match kept {
            Some(kept) => {
                assert!(ran.is_ok(), "{sql}: {ran:?}");
                assert_eq!(rows(&db), kept, "{sql}");
            }
            None => {
                let error = ran.unwrap_err().to_string();
                assert!(
                    error.contains("cannot affect a row a second time"),
                    "{sql}: {error}"
                );
                assert_eq!(rows(&db), [(1, 10), (2, 20)], "{sql} writes nothing");
            }
        }
    }
}

#[test]
fn a_row_a_cascade_wrote_is_not_met_twice() {
    let db = Database::open("memory://upsert_cascade_not_twice").unwrap();
    for sql in [
        "CREATE TABLE s (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, \
         parent INTEGER REFERENCES s(code) ON UPDATE CASCADE, n INTEGER)",
        "INSERT INTO s VALUES (1, 100, NULL, 0), (2, 200, 100, 0)",
        "INSERT INTO s VALUES (1, 100, NULL, 0), (2, 200, NULL, 0) \
         ON CONFLICT (id) DO UPDATE SET code = code + 1000, n = n + 1",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let left: Vec<(i64, i64, Option<i64>, i64)> = db
        .query("SELECT id, code, parent, n FROM s ORDER BY id", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(0).unwrap(),
                row.get::<i64>(1).unwrap(),
                row.get::<Option<i64>>(2).unwrap(),
                row.get::<i64>(3).unwrap(),
            )
        })
        .collect();
    assert_eq!(left, [(1, 1100, None, 1), (2, 1200, Some(1100), 1)]);
}
