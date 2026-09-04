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

/// A join inside a scalar subquery: the join's WHERE reads the row of the
/// query around it as well as its own rows
#[test]
fn test_join_where_reads_the_parent_row() {
    let db = setup("where_subquery_parent_row");
    // Users whose largest order is above u2.id * 10: every user for 1, users from 2 for 2, from 3 for 3
    let expected: Vec<Vec<String>> = vec![
        vec!["1".into(), "90".into()],
        vec!["2".into(), "87".into()],
        vec!["3".into(), "84".into()],
    ];
    assert_eq!(
        rows(
            &db,
            &format!(
                "SELECT u2.id, (SELECT COUNT(*) FROM {JOIN} WHERE (SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id) > u2.id * 10) FROM users u2 WHERE u2.id <= 3 ORDER BY u2.id"
            )
        ),
        expected
    );
}

/// A subquery inside LIKE or inside an IN list is resolved like a bare one
#[test]
fn test_subquery_inside_like_and_in_list() {
    let db = setup("where_subquery_like_list");
    // user1 and user10 to user19
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM users u WHERE u.name LIKE (SELECT name FROM users WHERE id = 1) || '%'"
        ),
        11,
        "LIKE over a subquery, scan"
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE u.name LIKE (SELECT name FROM users WHERE id = 1) || '%'")
        ),
        33,
        "LIKE over a subquery, join"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM users u WHERE u.id IN ((SELECT MIN(id) FROM users), 2)"
        ),
        2,
        "IN list holding a subquery, scan"
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE u.id IN ((SELECT MIN(id) FROM users), 2)")
        ),
        6,
        "IN list holding a subquery, join"
    );
}

/// The operand of a simple CASE may be a subquery
#[test]
fn test_subquery_as_case_operand() {
    let db = setup("where_subquery_case_operand");
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE CASE (SELECT MIN(id) FROM users) WHEN u.id THEN 1 ELSE 0 END = 1")
        ),
        3,
        "uncorrelated operand"
    );
    assert_eq!(
        rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {JOIN} WHERE CASE (SELECT MAX(amount) FROM orders x WHERE x.user_id = u.id) WHEN 302 THEN 1 ELSE 0 END = 1 GROUP BY u.id ORDER BY u.id")
        ),
        vec![vec!["30".to_string(), "3".to_string()]],
        "correlated operand"
    );
}

/// A correlated ALL in a join's WHERE: the largest order of every user
#[test]
fn test_correlated_all_in_join_where() {
    let db = setup("where_subquery_all_join");
    let predicate =
        "o.amount > ALL (SELECT amount FROM orders x WHERE x.user_id = u.id AND x.id <> o.id)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        30,
        "joined rows"
    );
    let expected: Vec<Vec<String>> = (1..=30i64)
        .map(|user| vec![user.to_string(), amount(user, 2).to_string()])
        .collect();
    assert_eq!(
        rows(
            &db,
            &format!("SELECT u.id, o.amount FROM {JOIN} WHERE {predicate} ORDER BY u.id")
        ),
        expected,
        "plain join"
    );
}

/// A subquery inside an IN list reaches the DML path too
#[test]
fn test_in_list_subquery_in_delete() {
    let db = setup("where_subquery_in_list_delete");
    db.execute(
        "DELETE FROM orders WHERE id IN ((SELECT MIN(id) FROM orders), 2)",
        (),
    )
    .unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM orders"), 88);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM orders WHERE id <= 2"), 0);
}

/// A column of the query around the join may sit inside a CAST or a CASE
#[test]
fn test_join_where_reads_the_parent_row_inside_a_container() {
    let db = setup("where_subquery_parent_row_container");
    let expected: Vec<Vec<String>> = vec![
        vec!["1".into(), "90".into()],
        vec!["2".into(), "87".into()],
        vec!["3".into(), "84".into()],
    ];
    assert_eq!(
        rows(
            &db,
            &format!(
                "SELECT u2.id, (SELECT COUNT(*) FROM {JOIN} WHERE (SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id) > CAST(u2.id AS FLOAT) * 10) FROM users u2 WHERE u2.id <= 3 ORDER BY u2.id"
            )
        ),
        expected,
        "CAST"
    );
    assert_eq!(
        rows(
            &db,
            &format!(
                "SELECT u2.id, (SELECT COUNT(*) FROM {JOIN} WHERE (SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id) > CASE WHEN u2.id = 1 THEN 10 WHEN u2.id = 2 THEN 20 ELSE 30 END) FROM users u2 WHERE u2.id <= 3 ORDER BY u2.id"
            )
        ),
        expected,
        "CASE"
    );
}

/// An outer reference wrapped in a CAST still makes the subquery correlated
#[test]
fn test_correlated_reference_inside_cast() {
    let db = setup("where_subquery_cast_reference");
    let predicate =
        "EXISTS (SELECT 1 FROM orders x WHERE x.user_id = CAST(u.id AS INTEGER) AND x.amount > 200)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        33,
        "joined rows"
    );
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
}

/// An uncorrelated subquery folds even next to a correlated one, so the
/// conjunct pushed to a derived table or CTE side is a plain predicate
#[test]
fn test_uncorrelated_next_to_correlated_on_a_derived_side() {
    let db = setup("where_subquery_mixed_derived");
    let predicate = "EXISTS (SELECT 1 FROM orders x WHERE x.user_id = u.id AND x.amount > 200) AND o.amount > (SELECT AVG(amount) FROM orders)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM users u INNER JOIN (SELECT * FROM orders) o ON u.id = o.user_id WHERE {predicate}")
        ),
        33,
        "derived table"
    );
    assert_eq!(
        count(
            &db,
            &format!("WITH o AS (SELECT * FROM orders) SELECT COUNT(*) FROM users u INNER JOIN o ON u.id = o.user_id WHERE {predicate}")
        ),
        33,
        "CTE"
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        33,
        "table"
    );
}

/// A column of the query around the join on the left of a correlated IN
#[test]
fn test_parent_column_in_a_correlated_in() {
    let db = setup("where_subquery_parent_in");
    let expected: Vec<Vec<String>> = (1..=3i64)
        .map(|user| vec![user.to_string(), "3".to_string()])
        .collect();
    assert_eq!(
        rows(
            &db,
            &format!(
                "SELECT u2.id, (SELECT COUNT(*) FROM {JOIN} WHERE u2.id IN (SELECT x.user_id FROM orders x WHERE x.user_id = u.id)) FROM users u2 WHERE u2.id <= 3 ORDER BY u2.id"
            )
        ),
        expected
    );
}

/// ALL and ANY over a set holding a NULL follow three-valued logic
#[test]
fn test_all_any_with_a_null_in_the_set() {
    let db = setup("where_subquery_all_null");
    db.execute("INSERT INTO orders VALUES (91, 1, NULL)", ())
        .unwrap();
    let cases = [
        (
            "200 > ALL (SELECT amount FROM orders WHERE user_id = 1)",
            0,
            "ALL that holds on every value but sees a NULL",
        ),
        (
            "200 > ALL (SELECT amount FROM orders WHERE user_id = 1 AND amount IS NOT NULL)",
            30,
            "ALL without the NULL",
        ),
        (
            "200 > ALL (SELECT amount FROM orders WHERE user_id = 99)",
            30,
            "ALL over an empty set",
        ),
        (
            "5 > ANY (SELECT amount FROM orders WHERE user_id = 1)",
            0,
            "ANY that fails on every value and sees a NULL",
        ),
        (
            "NOT (5 > ANY (SELECT amount FROM orders WHERE user_id = 1))",
            0,
            "NOT of an unknown ANY",
        ),
        (
            "11 > ANY (SELECT amount FROM orders WHERE user_id = 1)",
            30,
            "ANY that holds on one value",
        ),
        (
            "10 = ANY (SELECT amount FROM orders WHERE user_id = 1)",
            30,
            "= ANY that holds",
        ),
        (
            "NOT (7 = ANY (SELECT amount FROM orders WHERE user_id = 1))",
            0,
            "NOT of an unknown = ANY",
        ),
        (
            "7 <> ALL (SELECT amount FROM orders WHERE user_id = 1)",
            0,
            "<> ALL that sees a NULL",
        ),
    ];
    for (predicate, expected, label) in cases {
        assert_eq!(
            count(
                &db,
                &format!("SELECT COUNT(*) FROM users WHERE {predicate}")
            ),
            expected,
            "{label}"
        );
    }
    // The largest order of every user, except the user whose set holds a NULL
    let predicate =
        "o.amount > ALL (SELECT amount FROM orders x WHERE x.user_id = u.id AND x.id <> o.id)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        29,
        "correlated ALL"
    );
}

/// The left operand of ALL may itself hold a correlated subquery
#[test]
fn test_correlated_left_operand_of_all() {
    let db = setup("where_subquery_all_left");
    // Users whose largest order tops every order of user 20 (202): users from 21
    let predicate = "(SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id) > ALL (SELECT amount FROM orders WHERE user_id = 20)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        30,
        "joined rows"
    );
    let expected: Vec<Vec<String>> = (21..=30i64)
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
}

/// An outer column on the left of ALL is the only correlation of the
/// subquery: users from 21 clear every order of user 20
#[test]
fn test_outer_column_on_the_left_of_all_inside_a_subquery() {
    let db = setup("where_subquery_all_left_outer");
    let predicate = "EXISTS (SELECT 1 FROM orders x WHERE u.id * 10 > ALL (SELECT amount FROM orders WHERE user_id = 20) AND x.user_id = 25)";
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM users u WHERE {predicate}")
        ),
        10,
        "single table"
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT COUNT(*) FROM {JOIN} WHERE {predicate}")
        ),
        30,
        "joined rows"
    );
    let expected: Vec<Vec<String>> = (21..=30i64)
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
}

/// A subquery on the left of ALL in the SELECT list is resolved before the
/// expression compiles
#[test]
fn test_select_list_subquery_on_the_left_of_all() {
    let db = setup("select_subquery_all_left");
    assert_eq!(
        count(
            &db,
            "SELECT CASE WHEN (SELECT 2) > ALL (SELECT 1) THEN 1 ELSE 0 END"
        ),
        1,
        "scalar left of ALL"
    );
    assert_eq!(
        count(
            &db,
            "SELECT CASE WHEN (SELECT MAX(amount) FROM orders) > ALL (SELECT amount FROM orders WHERE user_id = 20) THEN 1 ELSE 0 END"
        ),
        1,
        "aggregate left of ALL"
    );
}

/// A subquery inside an aggregate's FILTER or argument is resolved before
/// the aggregates are parsed
#[test]
fn test_subquery_inside_an_aggregate() {
    let db = setup("select_subquery_in_aggregate");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FILTER (WHERE (SELECT 1) = 1) FROM users"
        ),
        30,
        "constant subquery in FILTER"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FILTER (WHERE amount > (SELECT AVG(amount) FROM orders)) FROM orders"
        ),
        orders_above_average(),
        "subquery in FILTER"
    );
    assert_eq!(
        count(
            &db,
            "SELECT SUM(CASE WHEN amount > (SELECT AVG(amount) FROM orders) THEN 1 ELSE 0 END) FROM orders"
        ),
        orders_above_average(),
        "subquery in an argument"
    );
    let avg = average_amount();
    let expected: Vec<Vec<String>> = (1..=30i64)
        .map(|user| {
            let n = (0..3i64).filter(|&k| amount(user, k) > avg).count();
            vec![user.to_string(), n.to_string()]
        })
        .collect();
    assert_eq!(
        rows(
            &db,
            "SELECT user_id, COUNT(*) FILTER (WHERE amount > (SELECT AVG(amount) FROM orders)) FROM orders GROUP BY user_id ORDER BY user_id"
        ),
        expected,
        "subquery in FILTER, grouped"
    );
}

/// ALL, BETWEEN, IN and LIKE in the SELECT list resolve the subqueries
/// they hold, with and without an outer row
#[test]
fn test_select_list_containers_hold_subqueries() {
    let db = setup("select_subquery_containers");
    for (select, label) in [
        (
            "CASE WHEN (SELECT 1) BETWEEN 0 AND 2 THEN 1 ELSE 0 END",
            "BETWEEN",
        ),
        (
            "CASE WHEN (SELECT 1) IN (1, 2) THEN 1 ELSE 0 END",
            "IN list",
        ),
        (
            "CASE WHEN 1 IN ((SELECT 1), 2) THEN 1 ELSE 0 END",
            "subquery in the IN list",
        ),
        (
            "CASE WHEN (SELECT 'ab') LIKE 'a%' THEN 1 ELSE 0 END",
            "LIKE",
        ),
    ] {
        assert_eq!(count(&db, &format!("SELECT {select}")), 1, "{label}");
    }
    let expected: Vec<Vec<String>> = vec![
        vec!["1".into(), "1".into(), "1".into()],
        vec!["2".into(), "1".into(), "1".into()],
        vec!["3".into(), "1".into(), "0".into()],
    ];
    assert_eq!(
        rows(
            &db,
            "SELECT u.id, CASE WHEN (SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id) > ALL (SELECT 0) THEN 1 ELSE 0 END, CASE WHEN (SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id) BETWEEN 10 AND 22 THEN 1 ELSE 0 END FROM users u WHERE u.id <= 3 ORDER BY u.id"
        ),
        expected,
        "correlated ALL and BETWEEN"
    );
}

/// An EXISTS, an ALL or a correlated subquery inside an aggregate is
/// resolved per input row before the aggregate runs
#[test]
fn test_every_subquery_kind_inside_an_aggregate() {
    let db = setup("select_subquery_kinds_in_aggregate");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1)) FROM users"
        ),
        30,
        "EXISTS in FILTER"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FILTER (WHERE 5 > ALL (SELECT 1)) FROM users"
        ),
        30,
        "ALL in FILTER"
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM orders x WHERE x.user_id = u.id AND x.amount > 200)) FROM users u"
        ),
        11,
        "correlated EXISTS in FILTER"
    );
    // The largest order of every user, summed: 10 * (1 + ... + 30) + 2 * 30
    assert_eq!(
        count(
            &db,
            "SELECT SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)) FROM users u"
        ),
        4710,
        "correlated scalar as an argument"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT u.id, SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)) FROM users u GROUP BY u.id ORDER BY u.id LIMIT 3"
        ),
        vec![
            vec!["1".to_string(), "12".to_string()],
            vec!["2".to_string(), "22".to_string()],
            vec!["3".to_string(), "32".to_string()],
        ],
        "correlated scalar as an argument, grouped"
    );
}

/// The aggregate subquery fix reaches a CTE and a recursive CTE
#[test]
fn test_subquery_inside_an_aggregate_over_a_cte() {
    let db = setup("select_subquery_in_aggregate_cte");
    assert_eq!(
        count(
            &db,
            "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FILTER (WHERE amount > (SELECT AVG(amount) FROM orders)) FROM t"
        ),
        orders_above_average(),
        "CTE"
    );
    let avg = average_amount();
    let expected: Vec<Vec<String>> = (1..=30i64)
        .map(|user| {
            let n = (0..3i64).filter(|&k| amount(user, k) > avg).count();
            vec![user.to_string(), n.to_string()]
        })
        .collect();
    assert_eq!(
        rows(
            &db,
            "WITH t AS (SELECT * FROM orders) SELECT user_id, COUNT(*) FILTER (WHERE amount > (SELECT AVG(amount) FROM orders)) FROM t GROUP BY user_id ORDER BY user_id"
        ),
        expected,
        "CTE, grouped"
    );
    assert_eq!(
        count(
            &db,
            "WITH RECURSIVE n(v) AS (SELECT 1 UNION ALL SELECT v + 1 FROM n WHERE v < 30) SELECT COUNT(*) FILTER (WHERE v > (SELECT MIN(id) FROM users) + 9) FROM n"
        ),
        20,
        "recursive CTE"
    );
    // The subquery inside the aggregate reads the CTE itself: orders above 150
    assert_eq!(
        count(
            &db,
            "WITH lim AS (SELECT 150 AS v) SELECT COUNT(*) FILTER (WHERE amount > (SELECT v FROM lim)) FROM orders"
        ),
        47,
        "aggregate subquery over the CTE"
    );
    // The correlated subquery reads the recursive CTE's column by its name
    assert_eq!(
        count(
            &db,
            "WITH RECURSIVE n(v) AS (SELECT 1 UNION ALL SELECT v + 1 FROM n WHERE v < 3) SELECT SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = n.v)) FROM n"
        ),
        66,
        "qualified CTE column inside the aggregate's subquery"
    );
}

/// An aggregate in HAVING or ORDER BY resolves the subquery it holds, and
/// so does the ORDER BY inside an aggregate call
#[test]
fn test_subquery_inside_an_aggregate_in_having_and_order_by() {
    let db = setup("select_subquery_in_having_order_by");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM (SELECT user_id FROM orders GROUP BY user_id HAVING COUNT(*) FILTER (WHERE EXISTS (SELECT 1)) > 2) t"
        ),
        30,
        "HAVING"
    );
    // Users 16 to 30 have all three orders above the average
    assert_eq!(
        rows(
            &db,
            "SELECT user_id FROM orders GROUP BY user_id ORDER BY COUNT(*) FILTER (WHERE amount > (SELECT AVG(amount) FROM orders)) DESC, user_id LIMIT 2"
        ),
        vec![vec!["16".to_string()], vec!["17".to_string()]],
        "ORDER BY"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT GROUP_CONCAT(name ORDER BY (SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id) DESC) FROM users u WHERE u.id <= 3"
        ),
        vec![vec!["user3,user2,user1".to_string()]],
        "ORDER BY inside the aggregate"
    );
}

/// A subquery inside an aggregate next to a window function, and inside a
/// window aggregate, is resolved before either runs
#[test]
fn test_subquery_inside_an_aggregate_with_a_window() {
    let db = setup("select_subquery_window");
    assert_eq!(
        rows(
            &db,
            "SELECT SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)), ROW_NUMBER() OVER () FROM users u"
        ),
        vec![vec!["4710".to_string(), "1".to_string()]],
        "aggregate beside a window function"
    );
    let windowed = rows(&db, "SELECT SUM((SELECT 1)) OVER () FROM users");
    assert_eq!(windowed.len(), 30, "window aggregate rows");
    assert!(
        windowed.iter().all(|row| row[0] == "30"),
        "window aggregate over a constant subquery: {windowed:?}"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT u.id, SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)) OVER () FROM users u WHERE u.id <= 3 ORDER BY u.id"
        ),
        vec![
            vec!["1".to_string(), "66".to_string()],
            vec!["2".to_string(), "66".to_string()],
            vec!["3".to_string(), "66".to_string()],
        ],
        "window aggregate over a correlated subquery"
    );
}

/// The column a lifted subquery lands in never takes a user column's name
#[test]
fn test_lifted_column_name_avoids_user_columns() {
    let db = Database::open("memory://lifted_column_name").unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY, __agg_subquery_0 INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO c VALUES (1, 100), (2, 200)", ())
        .unwrap();
    assert_eq!(
        count(&db, "SELECT SUM((SELECT 1)) + SUM(__agg_subquery_0) FROM c"),
        302
    );
    assert_eq!(
        count(
            &db,
            "SELECT SUM((SELECT MAX(id) FROM c d WHERE d.id = c.id)) + SUM(__agg_subquery_0) FROM c"
        ),
        303
    );
}
