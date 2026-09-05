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

/// An aggregate that reads the parent row is answered per parent row, not
/// once for the whole table with the first row's value inside it
#[test]
fn test_outer_reference_inside_an_aggregate_with_a_correlation() {
    let db = setup("correlated_scope_batch");
    // User n has three orders, each carrying user_id n
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT SUM(x.user_id * u.id) FROM orders x WHERE x.user_id = u.id) FROM users u ORDER BY u.id"
        ),
        ["3", "12", "27", "48"]
    );
    // Amounts of user n sum to 30n + 3, plus the parent id once per order
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT SUM(x.amount + u.id) FROM orders x WHERE x.user_id = u.id) FROM users u ORDER BY u.id"
        ),
        ["36", "69", "102", "135"]
    );
}

/// An alias over the parent's column keeps the alias and the parent's value
#[test]
fn test_outer_reference_under_an_alias() {
    let db = setup("correlated_scope_alias");
    assert_eq!(
        answers(
            &db,
            "SELECT u.id, (SELECT u.id AS parent_id FROM orders x LIMIT 1) FROM users u ORDER BY u.id"
        ),
        ["1", "2", "3", "4"]
    );
}

/// Two aggregates of one name over one table and one correlation column
/// read the columns they name, not one another's
#[test]
fn test_two_aggregates_over_one_correlation_column() {
    let db = setup("correlated_scope_two_aggregates");
    let rows: Vec<Vec<String>> = db
        .query(
            "SELECT u.id, (SELECT SUM(x.amount) FROM orders x WHERE x.user_id = u.id), (SELECT SUM(x.id) FROM orders x WHERE x.user_id = u.id) FROM users u ORDER BY u.id",
            (),
        )
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..3).map(|i| row.get::<String>(i).unwrap()).collect()
        })
        .collect();
    // Amounts of user n are 10n, 10n+1, 10n+2; its order ids are 3n-2, 3n-1, 3n
    assert_eq!(
        rows,
        vec![
            vec!["1", "33", "6"],
            vec!["2", "63", "15"],
            vec!["3", "93", "24"],
            vec!["4", "123", "33"],
        ]
    );
}

/// A subquery that reads the parent row does not answer for the one beside it
#[test]
fn test_two_subqueries_keep_their_own_answers() {
    let db = setup("correlated_scope_neighbours");
    let rows: Vec<Vec<String>> = db
        .query(
            "SELECT u.id, (SELECT SUM(x.user_id * u.id) FROM orders x WHERE x.user_id = u.id), (SELECT u.id FROM orders x LIMIT 1) FROM users u ORDER BY u.id",
            (),
        )
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..3).map(|i| row.get::<String>(i).unwrap()).collect()
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            vec!["1", "3", "1"],
            vec!["2", "12", "2"],
            vec!["3", "27", "3"],
            vec!["4", "48", "4"],
        ]
    );
}
