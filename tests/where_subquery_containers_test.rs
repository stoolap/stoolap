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

//! A subquery nested inside a CAST, a function call or a CASE in a WHERE,
//! and a correlated EXISTS in a join's WHERE, must give the answer the
//! same predicate gives when written plainly. Every expected value is
//! computed from the generated data, not from another query path.

use stoolap::Database;

/// 30 users, 3 orders each with amount user * 10 + k, k in 0..3
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)", ())
        .unwrap();
    let mut id = 0;
    for user in 1..=30i64 {
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            (user, format!("user{user}")),
        )
        .unwrap();
        for k in 0..3i64 {
            id += 1;
            db.execute(
                "INSERT INTO orders VALUES ($1, $2, $3)",
                (id, user, (user * 10 + k) as f64),
            )
            .unwrap();
        }
    }
    db
}

fn amount(user: i64, k: i64) -> f64 {
    (user * 10 + k) as f64
}

fn average_amount() -> f64 {
    let mut sum = 0.0;
    for user in 1..=30i64 {
        for k in 0..3i64 {
            sum += amount(user, k);
        }
    }
    sum / 90.0
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<String>> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| {
                    row.get::<Option<String>>(i)
                        .unwrap()
                        .unwrap_or_else(|| "NULL".into())
                })
                .collect()
        })
        .collect()
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

/// Groups of users with at least one order above the average, with the
/// number of such orders
fn groups_above_average() -> Vec<Vec<String>> {
    let avg = average_amount();
    let mut groups = Vec::new();
    for user in 1..=30i64 {
        let n = (0..3i64).filter(|&k| amount(user, k) > avg).count();
        if n > 0 {
            groups.push(vec![user.to_string(), n.to_string()]);
        }
    }
    groups
}

fn orders_above_average() -> i64 {
    groups_above_average()
        .iter()
        .map(|g| g[1].parse::<i64>().unwrap())
        .sum()
}

const JOIN: &str = "users u INNER JOIN orders o ON u.id = o.user_id";

#[test]
fn test_subquery_inside_cast() {
    let db = setup("where_subquery_cast");
    let predicate = "amount > CAST((SELECT AVG(amount) FROM orders) AS FLOAT)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM orders WHERE {predicate}")
        ),
        orders_above_average(),
        "scan"
    );
    let grouped = format!(
        "SELECT u.id, COUNT(o.id) FROM {JOIN} WHERE o.{predicate} GROUP BY u.id ORDER BY u.id"
    );
    assert_eq!(rows(&db, &grouped), groups_above_average(), "grouped join");
}

#[test]
fn test_subquery_inside_case() {
    let db = setup("where_subquery_case");
    let predicate = "CASE WHEN o.amount > (SELECT AVG(amount) FROM orders) THEN 1 ELSE 0 END = 1";
    let expected = groups_above_average();
    assert_eq!(
        rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {JOIN} WHERE {predicate} GROUP BY u.id ORDER BY u.id")
        ),
        expected,
        "grouped join with ORDER BY"
    );
    let mut limited = rows(
        &db,
        &format!("SELECT u.id, COUNT(o.id) FROM {JOIN} WHERE {predicate} GROUP BY u.id LIMIT 1000"),
    );
    limited.sort_by_key(|g| g[0].parse::<i64>().unwrap());
    assert_eq!(limited, expected, "grouped join with LIMIT");
}

#[test]
fn test_correlated_subquery_inside_function() {
    let db = setup("where_subquery_function");
    // Users whose largest order is above 200: user * 10 + 2 > 200
    let expected: Vec<Vec<String>> = (20..=30i64)
        .map(|user| vec![user.to_string(), "3".to_string()])
        .collect();
    let predicate = "COALESCE((SELECT MAX(amount) FROM orders x WHERE x.user_id = u.id), 0) > 200";
    assert_eq!(
        rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {JOIN} WHERE {predicate} GROUP BY u.id ORDER BY u.id")
        ),
        expected,
        "grouped join"
    );
    assert_eq!(
        rows(
            &db,
            &format!("SELECT u.id, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) FROM users u WHERE {predicate} ORDER BY u.id")
        ),
        expected,
        "single table"
    );
}

#[test]
fn test_correlated_exists_in_join_where() {
    let db = setup("where_subquery_exists_join");
    let predicate = "EXISTS (SELECT 1 FROM orders x WHERE x.user_id = u.id AND x.amount > 200)";
    let expected: Vec<Vec<String>> = (20..=30i64)
        .map(|user| vec![user.to_string(), "3".to_string()])
        .collect();
    assert_eq!(
        rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {JOIN} WHERE {predicate} GROUP BY u.id ORDER BY u.id")
        ),
        expected,
        "grouped join"
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        33,
        "joined rows"
    );
    let plain = rows(
        &db,
        &format!("SELECT u.id, o.amount FROM {JOIN} WHERE {predicate} ORDER BY u.id, o.amount"),
    );
    assert_eq!(plain.len(), 33, "plain join");
    assert_eq!(plain[0], vec!["20".to_string(), "200".to_string()]);
    let limited = rows(
        &db,
        &format!("SELECT u.id, o.amount FROM {JOIN} WHERE {predicate} LIMIT 10"),
    );
    assert_eq!(limited.len(), 10, "plain join with LIMIT");
}

/// A correlated subquery that reads both sides of the join can only be
/// evaluated on the joined row: every order except the user's largest
#[test]
fn test_correlated_subquery_over_both_sides() {
    let db = setup("where_subquery_both_sides");
    let predicate =
        "EXISTS (SELECT 1 FROM orders x WHERE x.user_id = u.id AND x.amount > o.amount)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        60,
        "joined rows"
    );
    let expected: Vec<Vec<String>> = (1..=30i64)
        .map(|user| vec![user.to_string(), "2".to_string()])
        .collect();
    assert_eq!(
        rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {JOIN} WHERE {predicate} GROUP BY u.id ORDER BY u.id")
        ),
        expected,
        "grouped join"
    );
    let limited = rows(
        &db,
        &format!("SELECT u.id, o.amount FROM {JOIN} WHERE {predicate} LIMIT 10"),
    );
    assert_eq!(limited.len(), 10, "plain join with LIMIT");
}
