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

//! A correlated subquery reads the parent row wherever it names it, not
//! only in its WHERE. A name its own FROM does not define belongs to the
//! parent, even when an inner column happens to share it.

use stoolap::Database;

/// 4 users, 3 orders each, order amount user * 10 + k
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, city TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
        (),
    )
    .unwrap();
    let mut order_id = 0i64;
    for user in 1..=4i64 {
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            (user, if user % 2 == 0 { "a" } else { "b" }),
        )
        .unwrap();
        for k in 0..3i64 {
            order_id += 1;
            db.execute(
                "INSERT INTO orders VALUES ($1, $2, $3)",
                (order_id, user, user * 10 + k),
            )
            .unwrap();
        }
    }
    db
}

/// The second column of every row, the subquery's answer
fn answers(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            row.get::<Option<String>>(1)
                .unwrap()
                .unwrap_or_else(|| "NULL".into())
        })
        .collect()
}

#[test]
fn test_outer_reference_in_an_aggregate_filter() {
    let db = setup("correlated_scope_filter");
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT COUNT(*) FILTER (WHERE x.user_id = u.id) FROM orders x) FROM users u ORDER BY u.id"
        ),
        ["3", "3", "3", "3"]
    );
}

#[test]
fn test_outer_reference_in_an_aggregate_argument() {
    let db = setup("correlated_scope_argument");
    // Orders of users 1 and 2 sum their user_id to 9, times the parent id
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT SUM(x.user_id * u.id) FROM orders x WHERE x.user_id <= 2) FROM users u ORDER BY u.id"
        ),
        ["9", "18", "27", "36"]
    );
}

#[test]
fn test_outer_reference_in_a_plain_expression() {
    let db = setup("correlated_scope_expression");
    // User 1's orders all carry user_id 1, so the max is 10 + the parent id
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT MAX(x.user_id * 10 + u.id) FROM orders x WHERE x.user_id = 1) FROM users u ORDER BY u.id"
        ),
        ["11", "12", "13", "14"]
    );
}

/// The parent's column wins over an inner column of the same bare name
#[test]
fn test_outer_reference_beats_a_same_named_inner_column() {
    let db = setup("correlated_scope_shadowing");
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT u.id FROM orders x LIMIT 1) FROM users u ORDER BY u.id"
        ),
        ["1", "2", "3", "4"],
        "the subquery selects the parent's id, not the order's"
    );
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT MAX(CASE WHEN x.user_id = 1 THEN u.id ELSE 0 END) FROM orders x) FROM users u ORDER BY u.id"
        ),
        ["1", "2", "3", "4"],
        "inside a CASE"
    );
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT MAX(x.id) FROM orders x WHERE x.user_id = u.id) FROM users u ORDER BY u.id"
        ),
        ["3", "6", "9", "12"],
        "the inner id still means the order's"
    );
}

/// A HAVING reads the parent row, and a subquery that names it only there
/// is still correlated
#[test]
fn test_outer_reference_in_having() {
    let db = setup("correlated_scope_having");
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT COUNT(*) FROM orders x GROUP BY x.user_id HAVING x.user_id = u.id) FROM users u ORDER BY u.id"
        ),
        ["3", "3", "3", "3"]
    );
}

/// What already worked keeps working
#[test]
fn test_outer_reference_in_where_still_works() {
    let db = setup("correlated_scope_where");
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT COUNT(*) FROM orders x WHERE x.user_id = u.id) FROM users u ORDER BY u.id"
        ),
        ["3", "3", "3", "3"]
    );
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT COUNT(*) FROM orders x WHERE x.amount > u.id * 10) FROM users u ORDER BY u.id"
        ),
        ["11", "8", "5", "2"],
        "orders above ten times the parent id"
    );
}
