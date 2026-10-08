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

//! A GROUP BY expression selected under an alias can be built on by another
//! column of the same SELECT, as it can when it is selected without one.

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Vec<(i64, Option<i64>)> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(0).unwrap(),
                row.get::<Option<i64>>(1).unwrap(),
            )
        })
        .collect()
}

#[test]
fn test_aliased_group_key_reused_by_another_column() {
    let db = Database::open("memory://group_by_aliased_key_reuse").unwrap();
    db.execute("CREATE TABLE t (ts INTEGER, price FLOAT)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (0, 1.0), (60, 2.0), (300, 3.0), (360, 5.0)",
        (),
    )
    .unwrap();

    let expected = vec![(0, Some(0)), (1, Some(300))];
    assert_eq!(
        rows(
            &db,
            "SELECT ts / 300 AS id, (ts / 300) * 300 AS bucket FROM t GROUP BY ts / 300 ORDER BY id"
        ),
        expected
    );
    assert_eq!(
        rows(
            &db,
            "SELECT ts / 300 AS id, (ts / 300) * 300 AS ts FROM t GROUP BY ts / 300 ORDER BY id"
        ),
        expected
    );
    assert_eq!(
        rows(
            &db,
            "SELECT ts / 300, (ts / 300) * 300 AS bucket FROM t GROUP BY ts / 300 ORDER BY 1"
        ),
        expected
    );
}

#[test]
fn test_aliased_function_group_key_reused_by_another_column() {
    let db = Database::open("memory://group_by_aliased_function_key_reuse").unwrap();
    db.execute("CREATE TABLE t (name TEXT)", ()).unwrap();
    db.execute("INSERT INTO t VALUES ('ab'), ('ab'), ('cde')", ())
        .unwrap();

    let lengths: Vec<Option<i64>> = db
        .query(
            "SELECT upper(name) AS u, length(upper(name)) AS l FROM t GROUP BY upper(name) ORDER BY u",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<Option<i64>>(1).unwrap())
        .collect();
    assert_eq!(lengths, vec![Some(2), Some(3)]);
}
