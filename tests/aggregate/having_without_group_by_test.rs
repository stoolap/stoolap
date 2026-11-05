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

//! HAVING without GROUP BY judges the single aggregate row, and drops it
//! when it does not hold, on the counting and min/max shortcuts too.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE hx (id INTEGER PRIMARY KEY, b INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO hx VALUES (1, 10), (2, 11), (3, 50), (4, 30)",
        (),
    )
    .unwrap();
    db
}

fn rows(db: &Database, sql: &str) -> usize {
    db.query(sql, ()).unwrap().count()
}

#[test]
fn test_having_drops_the_single_row_when_it_does_not_hold() {
    let db = setup("having_no_group_by");
    assert_eq!(
        rows(&db, "SELECT COUNT(*) FROM hx HAVING COUNT(*) > 100"),
        0
    );
    assert_eq!(rows(&db, "SELECT COUNT(*) FROM hx HAVING COUNT(*) > 1"), 1);
    assert_eq!(rows(&db, "SELECT MAX(b) FROM hx HAVING MAX(b) > 100"), 0);
    assert_eq!(rows(&db, "SELECT MIN(b) FROM hx HAVING MIN(b) < 100"), 1);
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*) FROM (SELECT COUNT(*) AS c FROM hx HAVING COUNT(*) > 100) t WHERE c IS NOT NULL"
        ),
        1
    );
    let inner: i64 = db
        .query(
            "SELECT COUNT(*) FROM (SELECT COUNT(*) AS c FROM hx HAVING COUNT(*) > 100)",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(inner, 0);
}
