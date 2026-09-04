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
