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

//! IN over a list of expressions that are not constants compares the
//! items in turn, with IN's NULL rule.

use stoolap::Database;

#[test]
fn test_in_over_expressions() {
    let db = Database::open("memory://in_list_of_expressions").unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 1, 10), (2, 2, 3), (3, 3, 3), (4, NULL, 5), (5, 6, NULL)",
        (),
    )
    .unwrap();
    let count = |where_clause: &str| -> i64 {
        db.query(&format!("SELECT COUNT(*) FROM x WHERE {where_clause}"), ())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap()
    };
    // 2 in (3, 4, 2) and 3 in (3, 4, 2)
    assert_eq!(count("a IN (b, b + 1, b - 1)"), 2);
    assert_eq!(count("a NOT IN (b, b + 1, b - 1)"), 1);
    assert_eq!(count("a IN (b, NULL)"), 1);
    assert_eq!(count("a NOT IN (b, NULL)"), 0);
    assert_eq!(count("a IN (b * 2, 1)"), 1);
}

#[test]
fn test_the_value_is_evaluated_once() {
    let db = Database::open("memory://in_list_value_once").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b BOOLEAN)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, true), (2, false)", ())
        .unwrap();
    // A value judged against b and NOT b is in the list whatever it is,
    // unless it is rolled again for each item
    for _ in 0..200 {
        let hits: Vec<bool> = db
            .query("SELECT (RANDOM() < 0.5) IN (b, NOT b) FROM t", ())
            .unwrap()
            .map(|r| r.unwrap().get::<bool>(0).unwrap())
            .collect();
        assert_eq!(hits, [true, true]);
    }
    let nulls: Vec<Option<bool>> = db
        .query(
            "SELECT id IN (NULL, id + 0), id IN (NULL, 5), NULL IN (id) FROM t",
            (),
        )
        .unwrap()
        .flat_map(|r| {
            let r = r.unwrap();
            [0, 1, 2].map(|i| r.get::<Option<bool>>(i).unwrap())
        })
        .collect();
    assert_eq!(nulls, [Some(true), None, None, Some(true), None, None]);
}
