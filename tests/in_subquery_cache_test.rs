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

//! The non-correlated IN-subquery cache memoizes a shared hash set per
//! entry; DML on the inner table must invalidate both the values and
//! the memoized set.

use stoolap::Database;

#[test]
fn in_subquery_set_survives_repeats_and_invalidation() {
    let db = Database::open("memory://in_subq_set_cache").unwrap();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE inner_t (id INTEGER PRIMARY KEY, v INTEGER)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..200i64 {
        tx.execute("INSERT INTO users VALUES ($1, $2)", (i, i))
            .unwrap();
    }
    for i in 0..50i64 {
        tx.execute("INSERT INTO inner_t VALUES ($1, $2)", (i, i))
            .unwrap();
    }
    tx.commit().unwrap();

    let stmt = db
        .prepare("SELECT COUNT(*) FROM users WHERE id IN (SELECT id FROM inner_t WHERE v >= 0)")
        .unwrap();
    // First run populates the cache, repeats use the memoized set
    for _ in 0..3 {
        let c: i64 = stmt.query_one(()).unwrap();
        assert_eq!(c, 50);
    }

    // INSERT into the inner table must invalidate the cached set
    db.execute("INSERT INTO inner_t VALUES (100, 0)", ())
        .unwrap();
    let c: i64 = stmt.query_one(()).unwrap();
    assert_eq!(c, 51);
    let c: i64 = stmt.query_one(()).unwrap();
    assert_eq!(c, 51);

    // DELETE must invalidate as well
    db.execute("DELETE FROM inner_t WHERE id = 0", ()).unwrap();
    let c: i64 = stmt.query_one(()).unwrap();
    assert_eq!(c, 50);

    // NOT IN goes through the same shared set
    let c: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM users WHERE id NOT IN (SELECT id FROM inner_t WHERE v >= 0)",
            (),
        )
        .unwrap();
    assert_eq!(c, 150);
}
