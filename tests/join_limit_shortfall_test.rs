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

//! A join with LIMIT fetches its outer or left side in a limited chunk. The
//! chunk must grow until the limit is met, whatever drops rows after it:
//! outer rows without a match, a cross-table WHERE, HAVING, or a right-side
//! filter on a LEFT JOIN.

use stoolap::Database;

fn count(db: &Database, sql: &str) -> usize {
    db.query(sql, ()).unwrap().count()
}

/// 100 users; 1000 orders of which one in twenty points at a user, the rest
/// at ids that do not exist. With `late`, the matched users are 50..=99 so
/// the first left chunk of a GROUP BY reduction has no group at all.
fn setup(name: &str, late: bool) -> Database {
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
    for i in 1..=100i64 {
        db.execute("INSERT INTO users VALUES ($1, $2)", (i, format!("U{i}")))
            .unwrap();
    }
    for i in 1..=1000i64 {
        let uid = if i % 20 != 0 {
            100_000 + i
        } else if late {
            100 - i / 20
        } else {
            i / 20
        };
        db.execute("INSERT INTO orders VALUES ($1, $2, $3)", (i, uid, i as f64))
            .unwrap();
    }
    db
}

#[test]
fn test_inl_join_limit_keeps_fetching_outer_rows_without_a_match() {
    let db = setup("join_limit_inl_outer", false);
    assert_eq!(
        count(&db, "SELECT o.amount, u.name FROM orders o INNER JOIN users u ON o.user_id = u.id LIMIT 100"),
        50
    );
    assert_eq!(
        count(&db, "SELECT u.name FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount > 0 LIMIT 100"),
        50
    );
    assert_eq!(
        count(
            &db,
            "SELECT u.name FROM users u INNER JOIN orders o ON u.id = o.user_id LIMIT 30"
        ),
        30
    );
}

#[test]
fn test_inl_join_limit_counts_rows_after_the_cross_table_filter() {
    let db = setup("join_limit_cross_filter", false);
    assert_eq!(
        count(&db, "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE (u.id + o.id) % 2 = 0 LIMIT 10"),
        10
    );
    assert_eq!(
        count(&db, "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE (u.id + o.id) % 2 = 0"),
        25
    );
}

#[test]
fn test_group_by_limit_reduction_grows_the_left_chunk() {
    let db = setup("join_limit_group_by", true);
    assert_eq!(
        count(&db, "SELECT u.name, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 10"),
        10
    );
    assert_eq!(
        count(&db, "SELECT u.name, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name HAVING COUNT(o.id) >= 1 LIMIT 10"),
        10
    );
    assert_eq!(
        count(&db, "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.amount > 0 GROUP BY u.id, u.name LIMIT 10"),
        10
    );
    // every matched user is one group: the values must be complete, not partial
    let total: i64 = db
        .query("SELECT SUM(c) FROM (SELECT COUNT(o.id) AS c FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 50) t", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(total, 50);
}

/// A window function or DISTINCT runs on the grouped rows after the join
/// produced them, so the reduction must hand over every group.
#[test]
fn test_group_by_limit_reduction_stays_off_under_a_window_function() {
    let db = setup("join_limit_window", true);
    let totals: Vec<i64> = db
        .query(
            "SELECT u.name, COUNT(o.id), COUNT(*) OVER () AS total FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 10",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(2).unwrap())
        .collect();
    assert_eq!(totals.len(), 10);
    assert!(totals.iter().all(|&t| t == 50), "window saw {totals:?}");
}

#[test]
fn test_group_by_limit_reduction_stays_off_under_distinct() {
    let db = setup("join_limit_distinct", true);
    // the 50 matched users are 50..=99, so u.id / 10 takes five values
    assert_eq!(
        count(&db, "SELECT DISTINCT u.id / 10 FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name LIMIT 10"),
        5
    );
}

/// The reduction doubles its left chunk and turns it into a LIMIT literal;
/// a huge LIMIT must not overflow into a negative one.
#[test]
fn test_group_by_limit_reduction_survives_a_huge_limit() {
    let db = setup("join_limit_huge", true);
    assert_eq!(
        count(&db, "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name HAVING COUNT(o.id) >= 0 LIMIT 5000000000000000000"),
        100
    );
    assert_eq!(
        count(&db, "SELECT u.name, COUNT(o.id) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name HAVING COUNT(o.id) >= 1 LIMIT 5000000000000000000"),
        50
    );
}

/// The outer side is fetched chunk by chunk; the rows across the chunk
/// boundaries must be exactly the matching ones, none repeated or missing.
#[test]
fn test_inl_join_limit_chunks_neither_repeat_nor_skip_rows() {
    let db = setup("join_limit_chunk_boundaries", false);
    let mut ids: Vec<i64> = db
        .query(
            "SELECT o.id FROM orders o INNER JOIN users u ON o.user_id = u.id LIMIT 100",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (1..=50).map(|i| i * 20).collect::<Vec<i64>>());
    let mut ids: Vec<i64> = db
        .query(
            "SELECT o.id FROM orders o INNER JOIN users u ON o.user_id = u.id LIMIT 30",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(ids.len(), 30);
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 30);
}

/// A CTE or a view on the outer side comes back whole; the join must not
/// treat it as a chunk and read it again from the start.
#[test]
fn test_inl_join_limit_over_a_cte_or_view_outer_never_repeats_rows() {
    let db = setup("join_limit_cte_view_outer", false);
    db.execute(
        "CREATE VIEW first_orders AS SELECT * FROM orders WHERE id <= 100",
        (),
    )
    .unwrap();
    // exactly 100 outer rows, five of them with a user: the first chunk is
    // consumed and still short of the limit
    for sql in [
        "WITH w AS (SELECT * FROM orders WHERE id <= 100) SELECT w.id FROM w INNER JOIN users u ON w.user_id = u.id LIMIT 10",
        "SELECT f.id FROM first_orders f INNER JOIN users u ON f.user_id = u.id LIMIT 10",
    ] {
        let mut ids: Vec<i64> = db
            .query(sql, ())
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![20, 40, 60, 80, 100], "{sql}");
    }
}

/// A table function on the outer side cannot be fetched again from an
/// offset, so its first fetch must not be cut at the chunk.
#[test]
fn test_inl_join_limit_over_a_table_function_outer_reads_it_whole() {
    let db = Database::open("memory://join_limit_tvf_outer").unwrap();
    db.execute("CREATE TABLE sparse (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO sparse VALUES (150)", ()).unwrap();
    let ids: Vec<i64> = db
        .query(
            "SELECT g.value FROM generate_series(1, 200) g INNER JOIN sparse s ON g.value = s.id LIMIT 1",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(ids, vec![150]);
}

/// A temporal outer source hands back every row, so the join must not take it
/// for a full chunk and read it again.
#[test]
fn test_inl_join_limit_over_a_temporal_outer_never_repeats_rows() {
    let db = setup("join_limit_temporal_outer", false);
    db.execute(
        "CREATE TABLE first100 (id INTEGER PRIMARY KEY, user_id INTEGER)",
        (),
    )
    .unwrap();
    for i in 1..=100i64 {
        let uid = if i % 20 == 0 { i / 20 } else { 100_000 + i };
        db.execute("INSERT INTO first100 VALUES ($1, $2)", (i, uid))
            .unwrap();
    }
    let mut ids: Vec<i64> = db
        .query(
            "SELECT f.id FROM first100 AS OF TIMESTAMP '2099-01-01 00:00:00' f INNER JOIN users u ON f.user_id = u.id LIMIT 10",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![20, 40, 60, 80, 100]);
}

/// A correlated filter on the outer side takes a scan path that stops early
/// at the chunk without reporting it; the fetch must still bound and continue.
#[test]
fn test_inl_join_limit_over_a_correlated_outer_filter_keeps_continuing() {
    let db = setup("join_limit_correlated_outer", false);
    let mut ids: Vec<i64> = db
        .query(
            "SELECT o.id FROM orders o INNER JOIN users u ON o.user_id = u.id WHERE o.amount >= (SELECT MIN(o2.amount) FROM orders o2 WHERE o2.user_id = o.user_id) LIMIT 100",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, (1..=50).map(|i| i * 20).collect::<Vec<i64>>());
}

/// Inside an explicit transaction the outer side is fetched whole; the
/// answer must still be complete.
#[test]
fn test_inl_join_limit_inside_an_explicit_transaction() {
    let db = setup("join_limit_explicit_tx", false);
    db.execute("BEGIN", ()).unwrap();
    assert_eq!(
        count(&db, "SELECT o.amount, u.name FROM orders o INNER JOIN users u ON o.user_id = u.id LIMIT 100"),
        50
    );
    assert_eq!(
        count(
            &db,
            "SELECT u.name FROM users u INNER JOIN orders o ON u.id = o.user_id LIMIT 30"
        ),
        30
    );
    db.execute("COMMIT", ()).unwrap();
}
