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

//! A boolean handed to SUM or AVG counts as one or nought, so a condition
//! can be summed to count the rows it holds for, and averaged for the
//! share of them. A NULL is left out as it is for any other value.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, b BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 50, TRUE), (2, 150, FALSE), (3, NULL, NULL), (4, 250, TRUE)",
        (),
    )
    .unwrap();
    db
}

fn one(db: &Database, sql: &str) -> String {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get::<Option<String>>(0)
        .unwrap()
        .unwrap_or_else(|| "NULL".into())
}

#[test]
fn test_summing_a_condition_counts_the_rows_it_holds_for() {
    let db = setup("boolean_agg_sum");
    assert_eq!(one(&db, "SELECT SUM(v > 100) FROM t"), "2");
    assert_eq!(one(&db, "SELECT SUM(b) FROM t"), "2");
    assert_eq!(one(&db, "SELECT SUM(TRUE) FROM t"), "4");
    assert_eq!(
        one(&db, "SELECT SUM(v > 100) FROM t"),
        one(
            &db,
            "SELECT SUM(CASE WHEN v > 100 THEN 1 ELSE 0 END) FROM t"
        ),
        "the same count the spelled-out form gives"
    );
}

#[test]
fn test_averaging_a_condition_gives_the_share_of_rows() {
    let db = setup("boolean_agg_avg");
    // Three rows carry a boolean; two of them are true
    let share: f64 = db
        .query("SELECT AVG(b) FROM t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert!((share - 2.0 / 3.0).abs() < 1e-9, "{share}");
}

#[test]
fn test_a_null_boolean_is_left_out() {
    let db = setup("boolean_agg_null");
    assert_eq!(one(&db, "SELECT SUM(b) FROM t WHERE id = 3"), "NULL");
    assert_eq!(one(&db, "SELECT COUNT(b) FROM t"), "3");
}
