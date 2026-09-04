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

//! A GROUP BY + LIMIT join whose groups are the left rows streams its groups
//! through the right side's index. Every shape is checked against the same
//! query with an ORDER BY, which takes the general hash path.

use stoolap::Database;

/// 100 users, 400 orders: user k has k % 7 orders, amounts 1..; users with
/// id % 10 == 0 have none. Every third order is 'shipped'.
fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, active BOOLEAN)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT, status TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)", ())
        .unwrap();
    for i in 1..=100i64 {
        db.execute(
            "INSERT INTO users VALUES ($1, $2, $3)",
            (i, format!("U{i}"), i % 3 != 0),
        )
        .unwrap();
    }
    let mut id = 0i64;
    for user in 1..=100i64 {
        if user % 10 == 0 {
            continue;
        }
        for k in 0..(user % 7) {
            id += 1;
            let status = if id % 3 == 0 { "shipped" } else { "completed" };
            db.execute(
                "INSERT INTO orders VALUES ($1, $2, $3, $4)",
                (id, user, (user * 10 + k) as f64, status),
            )
            .unwrap();
        }
    }
    db
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

/// The streamed result (LIMIT, no ORDER BY) must equal the general result
/// (ORDER BY u.id, no LIMIT) once both are sorted, and its length the limit.
fn check(db: &Database, select: &str, from_where: &str, group_having: &str, limit: usize) {
    let streamed = format!("SELECT {select} FROM {from_where} {group_having} LIMIT {limit}");
    let general = format!("SELECT {select} FROM {from_where} {group_having} ORDER BY u.id");
    let mut got = rows(db, &streamed);
    let mut want = rows(db, &general);
    got.sort();
    want.sort();
    assert!(got.len() <= limit, "{streamed}: {} rows", got.len());
    assert_eq!(got.len(), want.len().min(limit), "{streamed}");
    for row in &got {
        assert!(want.contains(row), "{streamed}: unexpected {row:?}");
    }
    if got.len() == want.len() {
        assert_eq!(got, want, "{streamed}");
    }
}

#[test]
fn test_grouped_join_matches_the_general_path() {
    let db = setup("grouped_join_stream");
    let inner = "users u INNER JOIN orders o ON u.id = o.user_id";
    let left = "users u LEFT JOIN orders o ON u.id = o.user_id";
    check(
        &db,
        "u.name, COUNT(o.id)",
        inner,
        "GROUP BY u.id, u.name",
        1000,
    );
    check(
        &db,
        "u.name, COUNT(o.id)",
        inner,
        "GROUP BY u.id, u.name",
        10,
    );
    check(
        &db,
        "u.name, COUNT(o.id), COUNT(*)",
        left,
        "GROUP BY u.id, u.name",
        1000,
    );
    check(&db, "u.name, COUNT(o.id)", left, "GROUP BY u.id, u.name", 7);
    check(
        &db,
        "u.name, COUNT(DISTINCT o.status) AS kinds, SUM(o.amount) AS total, AVG(o.amount), MIN(o.amount), MAX(o.amount)",
        inner,
        "GROUP BY u.id, u.name",
        1000,
    );
    check(
        &db,
        "u.name, COUNT(o.id) AS orders",
        inner,
        "GROUP BY u.id, u.name HAVING COUNT(o.id) > 2",
        1000,
    );
    check(
        &db,
        "u.name, COUNT(o.id) AS orders",
        inner,
        "GROUP BY u.id, u.name HAVING COUNT(o.id) > 2",
        5,
    );
    check(
        &db,
        "u.name, SUM(o.amount)",
        &format!("{inner} WHERE u.active = true AND o.status = 'completed'"),
        "GROUP BY u.id, u.name HAVING SUM(o.amount) > 100",
        1000,
    );
    check(
        &db,
        "u.name, COUNT(o.id)",
        &format!("{inner} WHERE o.amount > u.id * 10"),
        "GROUP BY u.id, u.name",
        1000,
    );
    check(
        &db,
        "u.name, SUM(o.amount) / COUNT(*) AS mean",
        inner,
        "GROUP BY u.id, u.name",
        1000,
    );
    check(&db, "u.id, COUNT(o.id)", left, "GROUP BY u.id", 1000);
}

#[test]
fn test_grouped_join_offset_and_limit() {
    let db = setup("grouped_join_stream_offset");
    let all = rows(
        &db,
        "SELECT u.name, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name ORDER BY u.id",
    );
    let page = rows(
        &db,
        "SELECT u.name, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 10 OFFSET 60",
    );
    assert_eq!(page.len(), 10.min(all.len() - 60));
    for row in &page {
        assert!(all.contains(row), "unexpected {row:?}");
    }
}

#[test]
fn test_grouped_join_rollup_keeps_the_total_row() {
    let db = setup("grouped_join_rollup");
    let with_limit = rows(
        &db,
        "SELECT u.id, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY ROLLUP(u.id) LIMIT 1000",
    );
    let general = rows(
        &db,
        "SELECT u.id, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY ROLLUP(u.id) ORDER BY u.id",
    );
    let mut with_limit = with_limit;
    let mut general = general;
    with_limit.sort();
    general.sort();
    assert_eq!(with_limit, general);
    assert!(
        general.iter().any(|row| row[0] == "NULL"),
        "rollup total row"
    );
}

#[test]
fn test_grouped_join_where_subquery() {
    let db = setup("grouped_join_where_subquery");
    for from_where in [
        "users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount > (SELECT AVG(amount) FROM orders)",
        "users u INNER JOIN orders o ON u.id = o.user_id WHERE u.id IN (SELECT user_id FROM orders WHERE amount > 500)",
        "users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount + u.id > (SELECT AVG(amount) FROM orders)",
    ] {
        check(&db, "u.id, COUNT(o.id), SUM(o.amount)", from_where, "GROUP BY u.id", 1000);
        check(&db, "u.id, COUNT(o.id)", from_where, "GROUP BY u.id", 7);
    }
}

#[test]
fn test_limited_join_where_subquery() {
    let db = setup("limited_join_where_subquery");
    let general = rows(
        &db,
        "SELECT u.id, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount > (SELECT AVG(amount) FROM orders) ORDER BY u.id, o.amount",
    );
    let mut limited = rows(
        &db,
        "SELECT u.id, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount > (SELECT AVG(amount) FROM orders) LIMIT 1000",
    );
    limited.sort();
    let mut general = general;
    general.sort();
    assert_eq!(limited, general);
    let page = rows(
        &db,
        "SELECT u.id, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount > (SELECT AVG(amount) FROM orders) LIMIT 7",
    );
    assert_eq!(page.len(), 7.min(general.len()));
}

#[test]
fn test_grouped_join_where_nested_subquery() {
    let db = setup("grouped_join_where_nested_subquery");
    for from_where in [
        "users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount BETWEEN (SELECT MIN(amount) FROM orders) AND (SELECT AVG(amount) FROM orders)",
        "users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount IN (SELECT amount FROM orders WHERE amount > 900)",
    ] {
        let general = rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {from_where} GROUP BY u.id ORDER BY u.id"),
        );
        assert!(!general.is_empty(), "general path returns rows for {from_where}");
        check(&db, "u.id, COUNT(o.id), SUM(o.amount)", from_where, "GROUP BY u.id", 1000);
        check(&db, "u.id, COUNT(o.id)", from_where, "GROUP BY u.id", 7);
    }
}

#[test]
fn test_grouped_join_right_filters() {
    let db = setup("grouped_join_right_filters");
    let inner = "users u INNER JOIN orders o ON u.id = o.user_id";
    for (from_where, group) in [
        (
            format!("{inner} WHERE o.status IN ('completed', 'shipped') AND o.amount > 300"),
            "GROUP BY u.id, u.name HAVING COUNT(o.id) > 1",
        ),
        (
            format!("{inner} WHERE LENGTH(o.status) > 7"),
            "GROUP BY u.id, u.name",
        ),
        (
            format!("{inner} WHERE o.status = 'completed' AND o.amount > u.id * 10 + 1"),
            "GROUP BY u.id, u.name",
        ),
        (
            format!("{inner} WHERE o.status LIKE 'c%' AND u.id > 20"),
            "GROUP BY u.id",
        ),
        (
            format!("{inner} WHERE o.status <> 'shipped' OR o.amount < 100"),
            "GROUP BY u.id",
        ),
    ] {
        check(
            &db,
            "u.name, COUNT(o.id), SUM(o.amount)",
            &from_where,
            group,
            1000,
        );
        check(&db, "u.name, COUNT(o.id)", &from_where, group, 9);
    }
}

#[test]
fn test_grouped_join_right_filter_typed_literals() {
    let db = Database::open("memory://grouped_join_typed_literals").unwrap();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT, status TEXT, created_at TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)", ())
        .unwrap();
    for user in 1..=30i64 {
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            (user, format!("user{user}")),
        )
        .unwrap();
    }
    let mut id = 0;
    for user in 1..=30i64 {
        for k in 0..3i64 {
            id += 1;
            let status = if (user + k) % 2 == 0 { "c_1" } else { "cx1" };
            let created = if k == 0 {
                "2024-01-15 10:00:00"
            } else {
                "2024-08-15 10:00:00"
            };
            db.execute(
                &format!(
                    "INSERT INTO orders VALUES ({id}, {user}, {}, '{status}', TIMESTAMP '{created}')",
                    user * 10 + k
                ),
                (),
            )
            .unwrap();
        }
    }
    for from_where in [
        "users u INNER JOIN orders o ON u.id = o.user_id WHERE o.status LIKE 'c!_%' ESCAPE '!'",
        "users u INNER JOIN orders o ON u.id = o.user_id WHERE o.created_at > TIMESTAMP '2024-06-01 00:00:00'",
    ] {
        let mut general = rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {from_where} GROUP BY u.id ORDER BY u.id"),
        );
        assert!(!general.is_empty(), "general path returns rows for {from_where}");
        let mut streamed = rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {from_where} GROUP BY u.id LIMIT 1000"),
        );
        streamed.sort();
        general.sort();
        assert_eq!(streamed, general, "{from_where}");
        let page = rows(
            &db,
            &format!("SELECT u.id, COUNT(o.id) FROM {from_where} GROUP BY u.id LIMIT 5"),
        );
        assert_eq!(page.len(), 5.min(general.len()), "{from_where}");
    }
}

#[test]
fn test_pushdown_or_with_a_partial_child() {
    let db = Database::open("memory://pushdown_or_partial_child").unwrap();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT, status TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)", ())
        .unwrap();
    let mut expected_rows = 0i64;
    let mut expected_xor = 0i64;
    let mut expected_groups: Vec<Vec<String>> = Vec::new();
    let mut id = 0;
    for user in 1..=30i64 {
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            (user, format!("user{user}")),
        )
        .unwrap();
        let mut per_user = 0i64;
        for k in 0..3i64 {
            id += 1;
            let status = if (user + k) % 2 == 0 { "c_1" } else { "cx1" };
            let amount = (user * 10 + k) as f64 - 15.0;
            db.execute(
                "INSERT INTO orders VALUES ($1, $2, $3, $4)",
                (id, user, amount, status),
            )
            .unwrap();
            if (amount > 10.0 && status == "c_1") || amount < 0.0 {
                per_user += 1;
            }
            if (amount > 10.0 && status == "c_1") ^ (amount < 0.0) {
                expected_xor += 1;
            }
        }
        expected_rows += per_user;
        if per_user > 0 {
            expected_groups.push(vec![user.to_string(), per_user.to_string()]);
        }
    }
    let scanned: i64 = db
        .query(
            "SELECT COUNT(*) FROM orders WHERE (amount > 10 AND status LIKE 'c!_%' ESCAPE '!') OR amount < 0",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(
        scanned, expected_rows,
        "scan with the OR of a partial child"
    );
    let xor: i64 = db
        .query(
            "SELECT COUNT(*) FROM orders WHERE (amount > 10 AND status LIKE 'c!_%' ESCAPE '!') XOR amount < 0",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(xor, expected_xor, "scan with the XOR of a partial child");
    let mut streamed = rows(
        &db,
        "SELECT u.id, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE (o.amount > 10 AND o.status LIKE 'c!_%' ESCAPE '!') OR o.amount < 0 GROUP BY u.id LIMIT 1000",
    );
    streamed.sort();
    expected_groups.sort();
    assert_eq!(
        streamed, expected_groups,
        "grouped stream with the OR of a partial child"
    );
}
