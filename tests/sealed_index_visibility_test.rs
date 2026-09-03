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

//! Once rows are sealed into volumes, the hot secondary index no longer
//! knows them. Every executor path that probes an index directly must still
//! see those rows, both right after a checkpoint and after a reopen.

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

fn rows(db: &Database, sql: &str) -> usize {
    db.query(sql, ()).unwrap().count()
}

fn populate(db: &Database) {
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount FLOAT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)", ())
        .unwrap();
    for i in 1..=20i64 {
        db.execute(
            "INSERT INTO users VALUES ($1, $2)",
            (i, format!("User_{}", i)),
        )
        .unwrap();
    }
    for i in 1..=60i64 {
        db.execute(
            "INSERT INTO orders VALUES ($1, $2, $3)",
            (i, (i % 20) + 1, i as f64),
        )
        .unwrap();
    }
}

/// The same answers the hot table gives: 60 joined rows, 3 orders per user.
fn assert_index_paths_see_every_row(db: &Database, label: &str) {
    assert_eq!(
        rows(
            db,
            "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id"
        ),
        60,
        "{label}: index nested loop join"
    );
    assert_eq!(
        rows(
            db,
            "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id LIMIT 10"
        ),
        10,
        "{label}: index nested loop join with LIMIT"
    );
    assert_eq!(
        rows(
            db,
            "SELECT u.id FROM users u LEFT JOIN orders o ON u.id = o.user_id"
        ),
        60,
        "{label}: left join"
    );
    assert_eq!(
        count(db, "SELECT COUNT(*) FROM orders WHERE user_id IN (5, 7)"),
        6,
        "{label}: IN list on the indexed column"
    );
    assert_eq!(
        count(
            db,
            "SELECT SUM(c) FROM (SELECT (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS c FROM users u) t"
        ),
        60,
        "{label}: correlated COUNT through the index"
    );
    assert_eq!(
        count(
            db,
            "SELECT COUNT(*) FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id)"
        ),
        20,
        "{label}: correlated EXISTS through the index"
    );
    assert_eq!(
        count(
            db,
            "SELECT COUNT(*) FROM orders WHERE user_id IN (SELECT id FROM users WHERE id <= 5)"
        ),
        15,
        "{label}: IN subquery on the indexed column"
    );
}

#[test]
fn test_sealed_rows_stay_visible_to_index_probes() {
    let dir = std::env::temp_dir().join(format!("stoolap_sealed_index_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dsn = format!("file://{}", dir.display());
    {
        let db = Database::open(&dsn).expect("Failed to create database");
        populate(&db);
        assert_index_paths_see_every_row(&db, "before checkpoint");
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_index_paths_see_every_row(&db, "after checkpoint");
        db.execute("INSERT INTO orders VALUES (61, 5, 61.0)", ())
            .unwrap();
        assert_eq!(
            rows(
                &db,
                "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id"
            ),
            61,
            "after checkpoint and one hot insert"
        );
        db.execute("DELETE FROM orders WHERE id = 61", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).expect("Failed to reopen database");
    assert_index_paths_see_every_row(&db, "after reopen");
    db.close().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
