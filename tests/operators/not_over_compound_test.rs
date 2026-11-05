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

//! NOT over AND or OR follows three-valued logic on the storage filter too:
//! NOT (UNKNOWN AND TRUE) is UNKNOWN, and a row it holds for is not kept.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    // every pairing of a in (NULL, 1, 2) with b in (NULL, 5, 20)
    db.execute(
        "INSERT INTO x VALUES (1, NULL, NULL), (2, NULL, 5), (3, NULL, 20), (4, 1, NULL), (5, 1, 5), (6, 1, 20), (7, 2, NULL), (8, 2, 5), (9, 2, 20)",
        (),
    )
    .unwrap();
    db
}

fn count(db: &Database, where_clause: &str) -> i64 {
    db.query(&format!("SELECT COUNT(*) FROM x WHERE {where_clause}"), ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

#[test]
fn test_not_over_a_conjunction_with_nulls() {
    let db = setup("not_over_and");
    // held only where a = 1 AND b > 10 is false outright: a = 2 (three rows),
    // a = 1 with b = 5, and a NULL with b = 5
    assert_eq!(count(&db, "NOT (a = 1 AND b > 10)"), 5);
    assert_eq!(count(&db, "NOT (a = 1 AND b > 10) AND id > 0"), 5);
    assert_eq!(count(&db, "NOT (a = 1 AND b > 10) OR b = 20"), 7);
    assert_eq!(count(&db, "NOT (a IS NULL AND b > 10)"), 7);
}

#[test]
fn test_not_over_a_disjunction_with_nulls() {
    let db = setup("not_over_or");
    // held only where both sides are false outright: a = 2 with b = 5
    assert_eq!(count(&db, "NOT (a = 1 OR b > 10)"), 1);
    assert_eq!(count(&db, "NOT (a = 1 OR b > 10) AND id > 0"), 1);
}
