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

//! Comparing a NULL leaves the answer unknown, so the row is not kept. This
//! holds when both sides are the same column, when both are NULL, and for
//! the operators that carry an equality half.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, n INTEGER, m INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO s VALUES (1, 10, 10), (2, 20, NULL), (3, NULL, NULL), (4, NULL, 5)",
        (),
    )
    .unwrap();
    db
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect()
}

/// A column compared with itself keeps the rows where it holds a value
#[test]
fn test_a_column_equals_itself_only_where_it_has_a_value() {
    let db = setup("null_cmp_self");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n = n"), 2);
    assert_eq!(ids(&db, "SELECT id FROM s WHERE n = n ORDER BY id"), [1, 2]);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n <> n"), 0);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n <= n"), 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n >= n"), 2);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n < n"), 0);
}

/// The same reading holds when the comparison is written out in full
#[test]
fn test_two_nulls_are_not_equal() {
    let db = setup("null_cmp_literal");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE NULL = NULL"), 0);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n = NULL"), 0);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE NULL <> NULL"), 0);
}

/// The operators that carry an equality half read a NULL the same way as `=`
#[test]
fn test_less_equal_and_greater_equal_do_not_match_two_nulls() {
    let db = setup("null_cmp_ordered");
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n = m"), 1);
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM s WHERE n <= m"),
        1,
        "only the row holding two tens"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n >= m"), 1);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n > m"), 0);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM s WHERE n < m"), 0);
}

/// Subtracting a column from itself is NULL wherever the column is
#[test]
fn test_a_column_minus_itself_is_null_where_the_column_is() {
    let db = setup("null_cmp_subtract");
    assert_eq!(count(&db, "SELECT COUNT(n - n) FROM s"), 2);
    assert_eq!(
        count(&db, "SELECT SUM(n - n) FROM s"),
        0,
        "the two rows holding a value each contribute a zero"
    );
}

/// A set holding nothing matches nothing, so NOT IN over it holds for
/// every row, including the rows the column is NULL in
#[test]
fn test_in_and_not_in_over_an_empty_set() {
    let db = Database::open("memory://null_cmp_empty_set").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO a VALUES (1, 10), (2, 20), (3, NULL)", ())
        .unwrap();
    db.execute("INSERT INTO b VALUES (1, 100), (2, NULL)", ())
        .unwrap();

    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM a WHERE k NOT IN (SELECT v FROM b WHERE v > 9999)"
        ),
        3
    );
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM a WHERE k NOT IN (SELECT v FROM b WHERE v > 9999) ORDER BY id"
        ),
        [1, 2, 3]
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM a WHERE k IN (SELECT v FROM b WHERE v > 9999)"
        ),
        0
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM a WHERE k NOT IN (SELECT v FROM b)"
        ),
        0,
        "a set holding a NULL still leaves it unknown"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM a WHERE k NOT IN (SELECT v FROM b WHERE v IS NOT NULL)"
        ),
        2
    );
}
