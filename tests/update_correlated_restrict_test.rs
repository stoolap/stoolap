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

//! ON UPDATE RESTRICT holds for an UPDATE whose WHERE is a correlated
//! subquery the semi-join rewrite refused: the check binds each row the way
//! the update does, so it looks at the rows the update would touch.

use stoolap::Database;

#[test]
fn test_restrict_holds_for_a_correlated_where() {
    let db = Database::open("memory://update_correlated_restrict").unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, code INTEGER UNIQUE)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, code INTEGER REFERENCES p(code) ON UPDATE RESTRICT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE q (k INTEGER)", ()).unwrap();
    db.execute("INSERT INTO p VALUES (1, 1), (2, 5)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 1)", ()).unwrap();
    db.execute("INSERT INTO q VALUES (1)", ()).unwrap();

    let refused = db.execute(
        "UPDATE p SET code = 2 WHERE EXISTS (SELECT 1 FROM q WHERE q.k = p.code LIMIT 1)",
        (),
    );
    assert!(refused.is_err(), "the referenced code must not change");
    let codes: Vec<i64> = db
        .query("SELECT code FROM p ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(codes, [1, 5]);

    // a row nothing references still changes
    let changed = db
        .execute(
            "UPDATE p SET code = 6 WHERE EXISTS (SELECT 1 FROM q WHERE q.k + 4 = p.code LIMIT 1)",
            (),
        )
        .unwrap();
    assert_eq!(changed, 1);
}
