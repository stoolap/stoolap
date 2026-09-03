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

/// DISTINCT runs after projection, so an early exit must not cut rows before
/// it, and the paths that apply LIMIT and OFFSET themselves must leave the
/// pair to the caller when DISTINCT is present.
#[test]
fn test_distinct_keeps_limit_and_offset_after_deduplication() {
    let db = setup("offset_distinct_paths");
    for (sql, expect) in [
        ("SELECT DISTINCT user_id FROM orders WHERE amount > 0 LIMIT 10 OFFSET 45", 5),
        ("SELECT DISTINCT user_id FROM orders WHERE user_id IN (SELECT id FROM users) LIMIT 10 OFFSET 45", 5),
        ("SELECT DISTINCT u.id % 10 FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id) LIMIT 3 OFFSET 8", 2),
        ("SELECT DISTINCT user_id FROM orders WHERE user_id IN (SELECT id FROM users) LIMIT 10", 10),
    ] {
        assert_eq!(count(&db, sql), expect, "{sql}");
    }
}

/// LEAD looks past the last fetched row, so the index-order window fetch
/// must not stop at LIMIT + OFFSET rows.
#[test]
fn test_lead_sees_the_row_after_the_limit_window() {
    let db = setup("offset_lead_window");
    let rows: Vec<(i64, Option<i64>)> = db
        .query(
            "SELECT id, LEAD(id) OVER (ORDER BY id) FROM orders LIMIT 10 OFFSET 45",
            (),
        )
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(0).unwrap(),
                row.get::<Option<i64>>(1).unwrap(),
            )
        })
        .collect();
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[0], (46, Some(47)));
    assert_eq!(rows[9], (55, Some(56)));
}

/// With ORDER BY the default window frame is RANGE up to the current row,
/// which includes the current row's later peers. A fetch cut at LIMIT +
/// OFFSET rows can split the last peer group, so peer-sensitive functions
/// must not use it.
#[test]
fn test_range_frame_peers_survive_the_limit_window() {
    let db = Database::open("memory://offset_range_peers").unwrap();
    db.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, category INTEGER NOT NULL, amount FLOAT)",
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
    // row 45 in category order is the first row of category 23; its frame holds
    // every row of categories 1..=23, including its peer past the 45th row
    for sql in [
        "SELECT category, SUM(amount) OVER (ORDER BY category) FROM sales LIMIT 1 OFFSET 44",
        "SELECT category, SUM(amount) OVER (ORDER BY category RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM sales LIMIT 1 OFFSET 44",
    ] {
        let row = db.query(sql, ()).unwrap().next().unwrap().unwrap();
        assert_eq!(row.get::<i64>(0).unwrap(), 23, "{sql}");
        assert_eq!(row.get::<f64>(1).unwrap(), 1702.0, "{sql}");
    }
    let last: f64 = db
        .query(
            "SELECT LAST_VALUE(amount) OVER (ORDER BY category) FROM sales LIMIT 1 OFFSET 44",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(last, 73.0);
}

/// The index-order fetch follows one window's ORDER BY; every other window
/// in the SELECT must order the same way, or it would rank a cut subset.
#[test]
fn test_second_window_with_another_order_keeps_the_full_fetch() {
    let db = setup("offset_second_window_order");
    let row = db
        .query(
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS by_id, ROW_NUMBER() OVER (ORDER BY amount DESC) AS by_amount FROM orders LIMIT 10 OFFSET 45",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 46);
    assert_eq!(row.get::<i64>(1).unwrap(), 46);
    // amount 46 is the 55th largest of 100, not the 10th of a 55-row subset
    assert_eq!(row.get::<i64>(2).unwrap(), 55);
}

/// The index orders NULLs its own way; the window's ORDER BY may not. Rows
/// fetched in index order must not be taken as pre-sorted on a nullable
/// column, with or without a limit: a table without the index is the oracle.
#[test]
fn test_window_over_a_nullable_index_column_matches_the_unindexed_table() {
    let db = Database::open("memory://offset_nullable_window").unwrap();
    for table in ["events", "events_plain"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER)"),
            (),
        )
        .unwrap();
        for i in 1..=20i64 {
            db.execute(&format!("INSERT INTO {table} VALUES ($1, NULL)"), (i,))
                .unwrap();
        }
        for i in 21..=100i64 {
            db.execute(
                &format!("INSERT INTO {table} VALUES ($1, $2)"),
                (i, 200 - i),
            )
            .unwrap();
        }
    }
    db.execute("CREATE INDEX idx_events_k ON events(k) USING BTREE", ())
        .unwrap();
    let read = |sql: &str| -> Vec<(i64, Option<i64>, Option<i64>)> {
        db.query(sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (
                    row.get::<i64>(0).unwrap(),
                    row.get::<Option<i64>>(1).unwrap(),
                    row.get::<Option<i64>>(2).unwrap(),
                )
            })
            .collect()
    };
    for order in ["ASC", "DESC", "ASC NULLS FIRST", "DESC NULLS LAST"] {
        // aggregates take the pre-sorted rows as they come, ranking sorts itself
        for func in ["ROW_NUMBER()", "COUNT(*)", "SUM(k)"] {
            let oracle = read(&format!(
                "SELECT id, k, {func} OVER (ORDER BY k {order}) AS v FROM events_plain ORDER BY id"
            ));
            let indexed = read(&format!(
                "SELECT id, k, {func} OVER (ORDER BY k {order}) AS v FROM events ORDER BY id"
            ));
            assert_eq!(indexed, oracle, "{func} OVER (ORDER BY k {order})");
        }
        let full = read(&format!(
            "SELECT id, k, ROW_NUMBER() OVER (ORDER BY k {order}) AS rn FROM events"
        ));
        for off in [0, 15, 75] {
            let cut = read(&format!(
                "SELECT id, k, ROW_NUMBER() OVER (ORDER BY k {order}) AS rn FROM events LIMIT 10 OFFSET {off}"
            ));
            assert_eq!(
                cut,
                full[off..off + 10].to_vec(),
                "ORDER BY k {order} OFFSET {off}"
            );
        }
    }
    // the first NULL row sorts after the 80 non-null rows by default
    let rn: i64 = db
        .query(
            "SELECT id, ROW_NUMBER() OVER (ORDER BY k) AS rn FROM events ORDER BY id LIMIT 1",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(1)
        .unwrap();
    assert_eq!(rn, 81);
}
