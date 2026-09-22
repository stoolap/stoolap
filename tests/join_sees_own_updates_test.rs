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

//! An index join inside a transaction sees the transaction's own rows: a
//! row moved to another key, a row inserted, a row deleted, before the
//! shared index takes the change at commit.

use stoolap::Database;

fn orders_in(db: &Database) {
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, amount INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_orders_user ON orders(user_id)", ())
        .unwrap();
    let user = db.prepare("INSERT INTO users VALUES (?, ?)").unwrap();
    for id in 1..=60 {
        user.execute((id, "u")).unwrap();
    }
    let order = db.prepare("INSERT INTO orders VALUES (?, ?, ?)").unwrap();
    for id in 1..=2400i64 {
        order.execute((id, (id - 1) % 60 + 1, 10)).unwrap();
    }
}

fn triples(db: &Database, sql: &str) -> Vec<(i64, i64, i64)> {
    let mut rows: Vec<_> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<i64>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();
    rows.sort_unstable();
    rows
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut rows: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    rows.sort_unstable();
    rows
}

const GROUPED: &str = "SELECT u.id, COUNT(o.id), SUM(o.amount) FROM users u \
    INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id LIMIT 5";
const USER_2_BOUNDED: &str = "SELECT o.id FROM users u INNER JOIN orders o ON u.id = o.user_id \
    WHERE u.id = 2 LIMIT 100";
const USER_2_ALL: &str =
    "SELECT o.id FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE u.id = 2";
const USER_3_ALL: &str =
    "SELECT o.id FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE u.id = 3";

/// Inside the transaction the join charges a moved order to its new user,
/// finds an inserted order under its user, and no longer finds a deleted
/// one; after commit the answers are the same
#[test]
fn an_index_join_sees_the_rows_its_transaction_moved_inserted_and_deleted() {
    let db = Database::open("memory://join_own_updates").unwrap();
    orders_in(&db);
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "UPDATE orders SET user_id = 2, amount = 1000 WHERE id = 1",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO orders VALUES (2401, 3, 10)", ())
        .unwrap();
    db.execute("DELETE FROM orders WHERE id = 2", ()).unwrap();

    let check = |when: &str| {
        let grouped = triples(&db, GROUPED);
        assert_eq!(
            grouped[0],
            (1, 39, 390),
            "{when}: the old user lost the order"
        );
        assert_eq!(
            grouped[1],
            (2, 40, 1390),
            "{when}: the new user has it, less the deleted one"
        );
        assert_eq!(
            grouped[2],
            (3, 41, 410),
            "{when}: the inserted order counts"
        );
        let user_2 = ids(&db, USER_2_BOUNDED);
        assert!(
            user_2.contains(&1),
            "{when}: the bounded join finds the moved order"
        );
        assert!(
            !user_2.contains(&2),
            "{when}: the bounded join drops the deleted order"
        );
        assert_eq!(user_2.len(), 40);
        assert_eq!(
            ids(&db, USER_2_ALL).len(),
            40,
            "{when}: the unbounded join too"
        );
        assert!(
            ids(&db, USER_3_ALL).contains(&2401),
            "{when}: the inserted order is joined"
        );
    };
    check("inside the transaction");
    db.execute("COMMIT", ()).unwrap();
    check("after commit");
}
