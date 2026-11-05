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

//! An index join hands its outer side an inner filter on the join key only
//! when that filter reads no other column of the inner side

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE o (id INTEGER PRIMARY KEY, t_id INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1, 0), (2, 2, 2), (3, 3, 0)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 1, 1), (2, 2, 9), (3, 3, 3)", ())
        .unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
        })
        .collect()
}

const CASES: &[&str] = &[
    "SELECT t.id, o.id FROM t JOIN o ON t.id = o.id WHERE t.a = t.id AND o.v = o.id ORDER BY 1",
    "SELECT t.id, o.id FROM t JOIN o ON o.id = t.id WHERE t.a > 0 AND o.id = o.v ORDER BY 1",
    "SELECT t.id, o.id FROM t JOIN o ON t.id = o.id WHERE t.a > 0 AND o.v + 0 = o.id ORDER BY 1",
    "SELECT t.id, o.id FROM o JOIN t ON t.id = o.id WHERE t.a > 0 AND o.v = o.id ORDER BY 1",
    "SELECT u.id, o.id FROM u JOIN o ON u.id = o.id WHERE u.a = u.id AND o.v = o.id ORDER BY 1",
    "SELECT u.id, o.id FROM u JOIN o ON u.id = o.id WHERE u.a > 0 AND o.v = o.id ORDER BY 1",
];

#[test]
fn an_inner_filter_reading_another_column_stays_inner() {
    let db = open("join_key_filter_outer");
    for sql in CASES {
        assert_eq!(ids(&db, sql), vec![(1, 1), (3, 3)], "{sql}");
    }
}

#[test]
fn an_inner_filter_on_the_key_alone_still_answers() {
    let db = open("join_key_filter_outer_key");
    let sql = "SELECT t.id, o.id FROM t JOIN o ON t.id = o.id WHERE t.a > 0 AND o.id IN (1, 3) ORDER BY 1";
    assert_eq!(ids(&db, sql), vec![(1, 1), (3, 3)], "{sql}");
}
