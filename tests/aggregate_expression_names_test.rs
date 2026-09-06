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

//! Two aggregates over expressions that differ only in their parentheses
//! are two aggregates, named apart, and computed apart.

use stoolap::Database;

fn row(db: &Database, sql: &str) -> (Vec<String>, Vec<i64>) {
    let result = db.query(sql, ()).unwrap();
    let columns = result.columns().to_vec();
    let values = result
        .map(|r| r.unwrap())
        .next()
        .map(|r| (0..r.len()).map(|i| r.get::<i64>(i).unwrap()).collect())
        .unwrap();
    (columns, values)
}

#[test]
fn test_parentheses_keep_aggregates_apart() {
    let db = Database::open("memory://aggregate_expression_names").unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, 1, 10), (2, 2, 20), (3, 3, 30)",
        (),
    )
    .unwrap();
    let (columns, values) = row(&db, "SELECT SUM(a + b * 2), SUM((a + b) * 2) FROM x");
    assert_eq!(columns, ["SUM(a + b * 2)", "SUM((a + b) * 2)"]);
    assert_eq!(values, [126, 132]);
    let (columns, values) = row(&db, "SELECT SUM(-(a + b)), SUM(-a + b) FROM x");
    assert_eq!(columns, ["SUM(-(a + b))", "SUM(-a + b)"]);
    assert_eq!(values, [-66, 54]);
    let (_, values) = row(&db, "SELECT SUM(a + b % 7), SUM((a + b) % 7) FROM x");
    assert_eq!(values, [17, 10]);
    let (_, values) = row(&db, "SELECT SUM((a - b) - 1), SUM(a - (b - 1)) FROM x");
    assert_eq!(values, [-57, -51]);
}
