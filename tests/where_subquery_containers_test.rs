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

/// A lifted subquery keeps its text as the result column's name, an identical
/// subquery shares its column, and a user column with that very name, bare
/// or qualified through a join, is never shadowed
#[test]
fn test_lifted_column_keeps_the_subquery_text() {
    let db = setup("lifted_column_name");
    let result = db
        .query(
            "SELECT SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)), COUNT((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)) FROM users u",
            (),
        )
        .unwrap();
    let columns: Vec<String> = result.columns().iter().map(|c| c.to_string()).collect();
    assert_eq!(
        columns,
        [
            "SUM((SELECT MAX(x.amount) FROM orders AS x WHERE (x.user_id = u.id)))",
            "COUNT((SELECT MAX(x.amount) FROM orders AS x WHERE (x.user_id = u.id)))",
        ]
    );
    let values: Vec<Vec<String>> = result
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get::<String>(i).unwrap())
                .collect()
        })
        .collect();
    assert_eq!(values, vec![vec!["4710".to_string(), "30".to_string()]]);

    let text = "(SELECT MAX(id) FROM c AS e WHERE (e.id = c.id))";
    db.execute(
        &format!("CREATE TABLE c (id INTEGER PRIMARY KEY, \"{text}\" INTEGER)"),
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (1, 100), (2, 200)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1), (2)", ()).unwrap();
    assert_eq!(
        count(
            &db,
            &format!(
                "SELECT SUM((SELECT MAX(id) FROM c e WHERE e.id = c.id)) + SUM(\"{text}\") FROM c"
            )
        ),
        303
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT SUM((SELECT MAX(id) FROM c e WHERE e.id = c.id)) + SUM(\"{text}\") FROM c JOIN d ON d.id = c.id")
        ),
        303
    );
}

/// A subquery inside an aggregate next to a window function is resolved on
/// the input rows once, and the window stage reads the rewritten statement
#[test]
fn test_aggregate_with_window_lifts_once() {
    let db = setup("aggregate_window_once");
    // MAX amount per user is user * 10 + 2: even users 2430, odd users 2280
    assert_eq!(
        rows(
            &db,
            "SELECT u.id % 2, SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)), ROW_NUMBER() OVER (ORDER BY u.id % 2) FROM users u GROUP BY u.id % 2 ORDER BY 1"
        ),
        vec![
            vec!["0".to_string(), "2430".to_string(), "1".to_string()],
            vec!["1".to_string(), "2280".to_string(), "2".to_string()],
        ],
        "grouped aggregate beside a window function"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id AND u.name <> '')), ROW_NUMBER() OVER () FROM users u"
        ),
        vec![vec!["4710".to_string(), "1".to_string()]],
        "a parent column the aggregate rows do not carry"
    );
}

/// A window aggregate honours its FILTER and its own ORDER BY
#[test]
fn test_window_aggregate_filter_and_order_by() {
    let db = setup("window_aggregate_filter");
    let none = rows(
        &db,
        "SELECT COUNT(*) FILTER (WHERE (SELECT 0) = 1) OVER () FROM users",
    );
    assert_eq!(none.len(), 30);
    assert!(none.iter().all(|row| row[0] == "0"), "{none:?}");
    let sums = rows(
        &db,
        "SELECT SUM(id) FILTER (WHERE id > 27) OVER () FROM users",
    );
    assert!(sums.iter().all(|row| row[0] == "87"), "{sums:?}");
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*) FILTER (WHERE id % 2 = 0) OVER (ORDER BY id) FROM users WHERE id <= 6 ORDER BY id"
        ),
        ["0", "1", "1", "2", "2", "3"]
            .iter()
            .map(|v| vec![v.to_string()])
            .collect::<Vec<_>>(),
        "running filtered count"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT id, COUNT(*) FILTER (WHERE id > 3) OVER (PARTITION BY id % 2) FROM users WHERE id <= 6 ORDER BY id"
        ),
        [("1", "1"), ("2", "2"), ("3", "1"), ("4", "2"), ("5", "1"), ("6", "2")]
            .iter()
            .map(|(id, n)| vec![id.to_string(), n.to_string()])
            .collect::<Vec<_>>(),
        "filtered count per partition"
    );
    // Even users 2, 4 and 6 have MAX amounts 22, 42 and 62
    let correlated = rows(
        &db,
        "SELECT SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = u.id)) FILTER (WHERE u.id % 2 = 0) OVER () FROM users u WHERE u.id <= 6"
    );
    assert_eq!(correlated.len(), 6);
    assert!(
        correlated.iter().all(|row| row[0] == "126"),
        "filtered correlated window aggregate: {correlated:?}"
    );
    let ordered = rows(
        &db,
        "SELECT GROUP_CONCAT(name ORDER BY id DESC) OVER () FROM users WHERE id <= 3",
    );
    assert_eq!(ordered.len(), 3);
    assert!(
        ordered.iter().all(|row| row[0] == "user3,user2,user1"),
        "ordered window aggregate: {ordered:?}"
    );
}

/// A subquery in an OVER clause or a named window orders and partitions the
/// rows, and the column it lands in stays out of SELECT *
#[test]
fn test_subquery_in_over_clause() {
    let db = setup("subquery_in_over");
    let reversed: Vec<Vec<String>> = (1..=6)
        .map(|id| vec![id.to_string(), (7 - id).to_string()])
        .collect();
    assert_eq!(
        rows(
            &db,
            "SELECT id, ROW_NUMBER() OVER (ORDER BY (SELECT -u.id)) FROM users u WHERE u.id <= 6 ORDER BY id"
        ),
        reversed,
        "FROM-less subquery in ORDER BY"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT id, ROW_NUMBER() OVER (ORDER BY (SELECT -MAX(x.id) FROM users x WHERE x.id = u.id)) FROM users u WHERE u.id <= 6 ORDER BY id"
        ),
        reversed,
        "subquery with a FROM in ORDER BY"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT id, ROW_NUMBER() OVER w FROM users u WHERE u.id <= 6 WINDOW w AS (ORDER BY (SELECT -u.id)) ORDER BY id"
        ),
        reversed,
        "named window"
    );
    let counts = rows(
        &db,
        "SELECT id, COUNT(*) OVER (PARTITION BY (SELECT u.id % 2)) FROM users u WHERE u.id <= 6 ORDER BY id"
    );
    assert_eq!(counts.len(), 6);
    assert!(
        counts.iter().all(|row| row[1] == "3"),
        "subquery in PARTITION BY: {counts:?}"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT -u.id)) FROM users u WHERE u.id <= 2 ORDER BY id"
        ),
        vec![
            vec!["1".to_string(), "user1".to_string(), "2".to_string()],
            vec!["2".to_string(), "user2".to_string(), "1".to_string()],
        ],
        "SELECT * beside a lifted OVER clause"
    );
}

/// A correlated subquery beside a window function is read per input row,
/// also on the indexed partition fetch that LIMIT would take
#[test]
fn test_correlated_subquery_beside_a_window_function() {
    let db = setup("subquery_beside_window");
    let limited = rows(
        &db,
        "SELECT id, COUNT(*) OVER (PARTITION BY id) + (SELECT u.id * 0) FROM users u LIMIT 2",
    );
    assert_eq!(limited.len(), 2);
    assert!(
        limited.iter().all(|row| row[1] == "1"),
        "beside a partitioned count with LIMIT: {limited:?}"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT id, SUM((SELECT u.id)) OVER (PARTITION BY id % 2) FROM users u WHERE u.id <= 6 ORDER BY id"
        ),
        [("1", "9"), ("2", "12"), ("3", "9"), ("4", "12"), ("5", "9"), ("6", "12")]
            .iter()
            .map(|(id, sum)| vec![id.to_string(), sum.to_string()])
            .collect::<Vec<_>>(),
        "window aggregate over a FROM-less subquery"
    );
}

/// A FROM-less subquery in a WHERE reads the parent row
#[test]
fn test_fromless_correlated_subquery_in_where() {
    let db = setup("fromless_where");
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM users u WHERE (SELECT u.id) > 27"),
        3
    );
}

/// Two subqueries whose text is the same but whose anonymous parameters are
/// not each read their own binding, within one statement and across two
#[test]
fn test_subqueries_keep_their_own_parameter() {
    let db = setup("subquery_parameters");
    // MAX(x.amount) for user n is n * 10 + 2, summed over the 30 users
    let row = db
        .query(
            "SELECT SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = ?)), SUM((SELECT MAX(x.amount) FROM orders x WHERE x.user_id = ?)) FROM users u",
            (1i64, 2i64),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(
        (row.get::<String>(0).unwrap(), row.get::<String>(1).unwrap()),
        ("360".to_string(), "660".to_string()),
        "two anonymous parameters in one statement"
    );

    let one = "SELECT (SELECT MAX(x.amount) FROM orders x WHERE x.user_id = ?) FROM users LIMIT 1";
    let read = |p: i64| -> String {
        db.query(one, (p,))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get::<String>(0)
            .unwrap()
    };
    assert_eq!(
        (read(1), read(2)),
        ("12".to_string(), "22".to_string()),
        "one statement, two executions"
    );
}

/// A navigation window function reads its control argument off no row, so a
/// subquery there stays where it is instead of becoming a column
#[test]
fn test_navigation_window_control_argument() {
    let db = setup("navigation_control_argument");
    // An unqualified argument, since a qualified one is a separate bug that
    // predates this branch and shows the same way on main
    let ids = "WHERE id <= 4 ORDER BY id";
    assert_eq!(
        rows(
            &db,
            &format!("SELECT id, LAG(id, (SELECT 1)) OVER (ORDER BY id) FROM users {ids}")
        ),
        [("1", "NULL"), ("2", "1"), ("3", "2"), ("4", "3")]
            .iter()
            .map(|(a, b)| vec![a.to_string(), b.to_string()])
            .collect::<Vec<_>>(),
        "LAG offset"
    );
    assert_eq!(
        rows(
            &db,
            &format!("SELECT id, NTILE((SELECT 2)) OVER (ORDER BY id) FROM users {ids}")
        ),
        [("1", "1"), ("2", "1"), ("3", "2"), ("4", "2")]
            .iter()
            .map(|(a, b)| vec![a.to_string(), b.to_string()])
            .collect::<Vec<_>>(),
        "NTILE group count"
    );
    assert_eq!(
        rows(
            &db,
            &format!("SELECT id, NTH_VALUE(id, (SELECT 2)) OVER (ORDER BY id) FROM users {ids}")
        ),
        [("1", "NULL"), ("2", "2"), ("3", "2"), ("4", "2")]
            .iter()
            .map(|(a, b)| vec![a.to_string(), b.to_string()])
            .collect::<Vec<_>>(),
        "NTH_VALUE n"
    );
    // A correlated control argument reads no row either, so it keeps the
    // answer it has always given rather than failing to compile
    assert_eq!(
        rows(
            &db,
            "SELECT u.id, NTILE((SELECT MAX(x.user_id) FROM orders x WHERE x.user_id = u.id)) OVER (ORDER BY u.id) FROM users u WHERE u.id <= 4 ORDER BY u.id"
        )
        .len(),
        4,
        "correlated control argument still runs"
    );
}

/// An aggregate that orders its own input places NULLs where the query says
#[test]
fn test_aggregate_order_by_nulls_placement() {
    let db = Database::open("memory://aggregate_order_nulls").unwrap();
    db.execute(
        "CREATE TABLE n (id INTEGER PRIMARY KEY, k INTEGER, v TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO n VALUES (1, 2, 'x'), (2, NULL, 'y'), (3, 1, 'z')",
        (),
    )
    .unwrap();
    let first = |sql: &str| -> String { rows(&db, sql)[0][0].clone() };
    assert_eq!(
        first("SELECT GROUP_CONCAT(v ORDER BY k NULLS FIRST) FROM n"),
        "y,z,x"
    );
    assert_eq!(
        first("SELECT GROUP_CONCAT(v ORDER BY k NULLS LAST) FROM n"),
        "z,x,y"
    );
    assert_eq!(
        first("SELECT GROUP_CONCAT(v ORDER BY k DESC NULLS LAST) FROM n"),
        "x,z,y"
    );
    assert_eq!(
        first("SELECT GROUP_CONCAT(v ORDER BY k DESC) FROM n"),
        "y,x,z",
        "DESC without a NULLS clause puts them first"
    );
    assert_eq!(
        first("SELECT GROUP_CONCAT(v ORDER BY k NULLS FIRST) OVER () FROM n LIMIT 1"),
        "y,z,x",
        "the same inside a window"
    );
    assert_eq!(
        first("SELECT GROUP_CONCAT(v ORDER BY k DESC NULLS LAST) OVER () FROM n LIMIT 1"),
        "x,z,y",
        "descending inside a window"
    );
}
