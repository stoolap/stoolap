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

//! OFFSET skips rows before LIMIT counts them, and is applied exactly once,
//! on every path that stops early: GROUP BY, window fetches, subqueries in
//! WHERE and joins.

use stoolap::Database;

fn count(db: &Database, sql: &str) -> usize {
    db.query(sql, ()).unwrap().count()
}

/// 50 groups: user ids 1..=50, each with two orders.
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
    for i in 1..=50i64 {
        db.execute("INSERT INTO users VALUES ($1, $2)", (i, format!("U{i}")))
            .unwrap();
    }
    for i in 1..=100i64 {
        db.execute(
            "INSERT INTO orders VALUES ($1, $2, $3)",
            (i, (i - 1) % 50 + 1, i as f64),
        )
        .unwrap();
    }
    db
}

#[test]
fn test_group_by_offset_is_applied_before_limit() {
    let db = setup("group_by_offset_single");
    for (sql, expect) in [
        ("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id LIMIT 10 OFFSET 45", 5),
        ("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id LIMIT 10 OFFSET 5", 10),
        ("SELECT user_id, SUM(amount) FROM orders GROUP BY user_id ORDER BY user_id LIMIT 10 OFFSET 45", 5),
        ("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) = 2 LIMIT 10 OFFSET 45", 5),
        ("SELECT user_id, COUNT(*) FROM orders WHERE amount > 0 GROUP BY user_id LIMIT 10 OFFSET 45", 5),
        ("SELECT user_id FROM orders GROUP BY user_id OFFSET 45", 5),
    ] {
        assert_eq!(count(&db, sql), expect, "{sql}");
    }
}

#[test]
fn test_group_by_offset_is_applied_before_limit_on_joins() {
    let db = setup("group_by_offset_join");
    for (sql, expect) in [
        ("SELECT u.name, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 10 OFFSET 45", 5),
        ("SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 10 OFFSET 5", 10),
        ("SELECT u.name, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name HAVING COUNT(o.id) = 2 LIMIT 10 OFFSET 45", 5),
    ] {
        assert_eq!(count(&db, sql), expect, "{sql}");
    }
}

#[test]
fn test_offset_is_applied_before_limit_on_other_pushdown_paths() {
    let db = setup("offset_other_paths");
    for (sql, expect) in [
        ("SELECT id FROM users WHERE id > 10 ORDER BY id LIMIT 10 OFFSET 35", 5),
        ("SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM users LIMIT 10 OFFSET 45", 5),
        ("SELECT id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY id) FROM orders LIMIT 10 OFFSET 95", 5),
        ("WITH c AS (SELECT * FROM users) SELECT * FROM c LIMIT 10 OFFSET 45", 5),
        ("SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 0 LIMIT 10 OFFSET 45", 5),
        ("SELECT DISTINCT user_id FROM orders LIMIT 10 OFFSET 45", 5),
        ("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders) LIMIT 10 OFFSET 45", 5),
        ("SELECT u.name FROM users u INNER JOIN orders o ON u.id = o.user_id LIMIT 10 OFFSET 95", 5),
    ] {
        assert_eq!(count(&db, sql), expect, "{sql}");
    }
}

/// The streaming GROUP BY walks a BTREE index in order and stops early at
/// the limit; the groups OFFSET skips must be walked too.
#[test]
fn test_streaming_group_by_offset_is_applied_before_limit() {
    let db = Database::open("memory://streaming_group_by_offset").unwrap();
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, category INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE INDEX idx_sales_category ON sales(category) USING BTREE",
        (),
    )
    .unwrap();
    for i in 1..=100i64 {
        db.execute(
            "INSERT INTO sales VALUES ($1, $2, $3)",
            (i, (i - 1) % 50 + 1, i as f64),
        )
        .unwrap();
    }
    for (sql, expect) in [
        ("SELECT category, SUM(amount) FROM sales GROUP BY category LIMIT 10 OFFSET 45", 5),
        ("SELECT category, COUNT(*) FROM sales GROUP BY category LIMIT 10 OFFSET 5", 10),
        ("SELECT category, COUNT(*) FROM sales GROUP BY category HAVING COUNT(*) = 2 LIMIT 10 OFFSET 45", 5),
    ] {
        assert_eq!(count(&db, sql), expect, "{sql}");
    }
}

/// A subquery in WHERE takes fast paths that apply LIMIT and OFFSET on their
/// own; they must skip the OFFSET rows and report the pair as applied so the
/// caller does not skip them again.
#[test]
fn test_subquery_where_offset_is_applied_once() {
    let db = setup("offset_subquery_where");
    let ids: Vec<String> = (1..=50).map(|i| i.to_string()).collect();
    let in_list = format!(
        "SELECT * FROM users WHERE id IN ({}) LIMIT 10 OFFSET 45",
        ids.join(",")
    );
    for (sql, expect) in [
        (in_list.as_str(), 5),
        ("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders) LIMIT 10 OFFSET 45", 5),
        ("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders) OFFSET 45", 5),
        ("SELECT * FROM orders WHERE user_id IN (SELECT id FROM users) LIMIT 10 OFFSET 95", 5),
        ("SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM orders WHERE user_id > 40) LIMIT 10 OFFSET 35", 5),
        ("SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id) LIMIT 10 OFFSET 45", 5),
        ("SELECT * FROM users WHERE id > (SELECT MIN(id) FROM users) LIMIT 10 OFFSET 44", 5),
        ("SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders) LIMIT 10 OFFSET 45", 5),
    ] {
        assert_eq!(count(&db, sql), expect, "{sql}");
    }
    let names: Vec<String> = db
        .query(
            "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders) ORDER BY name DESC LIMIT 3",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(names, ["U9", "U8", "U7"]);
}
