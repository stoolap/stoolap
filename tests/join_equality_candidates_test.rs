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

//! A limited join against a persistent table whose rows are all still in
//! memory asks the table for the ids of each key instead of scanning it.
//! The table answers only while no volume holds rows, no probe returns more
//! than the cap, and nothing moved its index across the probe; a join it
//! stops answering runs once more on the hash path.

use stoolap::Database;

const SELF_JOIN: &str = "SELECT u1.id, u2.id, u1.age FROM users u1 \
    INNER JOIN users u2 ON u1.age = u2.age AND u1.id < u2.id LIMIT 100";

fn file_db(dir: &tempfile::TempDir) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap()
}

/// `rows` users whose age is 18 plus the id modulo 60, named by parity
fn users_in(db: &Database, rows: i64) {
    db.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_users_age ON users(age)", ())
        .unwrap();
    let insert = db.prepare("INSERT INTO users VALUES (?, ?, ?)").unwrap();
    for id in 1..=rows {
        let name = if id % 2 == 0 { "even" } else { "odd" };
        insert.execute((id, name, 18 + id % 60)).unwrap();
    }
}

/// A small outer table whose ages are given
fn people_in(db: &Database, ages: &[i64]) {
    db.execute(
        "CREATE TABLE people (id INTEGER PRIMARY KEY, age INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    let insert = db.prepare("INSERT INTO people VALUES (?, ?)").unwrap();
    for (i, age) in ages.iter().enumerate() {
        insert.execute((i as i64 + 1, *age)).unwrap();
    }
}

fn triples(db: &Database, sql: &str) -> Vec<(i64, Option<i64>, i64)> {
    let mut rows: Vec<_> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<Option<i64>>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();
    rows.sort_unstable();
    rows
}

#[cfg(feature = "test-failpoints")]
mod probes {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Counts every probe admitted to a hot index on this thread until
    /// `stop` is called
    pub fn count(counter: Arc<AtomicUsize>) {
        stoolap::test_failpoints::after_join_probe_admitted(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            count(counter);
        });
    }

    pub fn stop() {
        stoolap::test_failpoints::after_join_probe_admitted(|| {});
    }
}

#[test]
fn a_limited_self_join_on_a_hot_file_table_matches_the_memory_engine() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    users_in(&db, 2000);
    let oracle = Database::open("memory://join_eq_oracle_self").unwrap();
    users_in(&oracle, 2000);

    let expected = triples(&oracle, SELF_JOIN);
    assert_eq!(expected.len(), 100);
    assert_eq!(triples(&db, SELF_JOIN), expected, "hot rows");

    // Sealed rows are joined on the hash path, which picks its own hundred
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let every_pair = triples(&oracle, SELF_JOIN.trim_end_matches(" LIMIT 100"));
    let sealed = triples(&db, SELF_JOIN);
    assert_eq!(sealed.len(), 100, "sealed rows");
    assert!(sealed.iter().all(|pair| every_pair.contains(pair)));
}

/// An outer key no inner row carries joins nothing, and a LEFT join pads it
#[test]
fn an_outer_key_without_inner_rows_joins_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    users_in(&db, 600);
    people_in(&db, &[18, 200, 19]);
    let oracle = Database::open("memory://join_eq_oracle_empty").unwrap();
    users_in(&oracle, 600);
    people_in(&oracle, &[18, 200, 19]);

    let inner = "SELECT p.id, u.id, p.age FROM people p \
        INNER JOIN users u ON p.age = u.age LIMIT 1000";
    let got = triples(&db, inner);
    assert_eq!(got, triples(&oracle, inner));
    assert_eq!(got.len(), 20, "ten users of each present age");
    assert!(got.iter().all(|(_, _, age)| *age != 200));

    let left = "SELECT p.id, u.id, p.age FROM people p \
        LEFT JOIN users u ON p.age = u.age LIMIT 1000";
    let got = triples(&db, left);
    assert_eq!(got, triples(&oracle, left));
    assert!(got.contains(&(2, None, 200)), "the absent key is padded");
}

/// The rest of the ON clause is applied to the rows the probe found
#[test]
fn the_rest_of_the_on_clause_filters_the_probed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    users_in(&db, 2000);
    let oracle = Database::open("memory://join_eq_oracle_residual").unwrap();
    users_in(&oracle, 2000);

    let sql = "SELECT u1.id, u2.id, u1.age FROM users u1 \
        INNER JOIN users u2 ON u1.age = u2.age AND u1.id < u2.id AND u2.name = 'even' \
        LIMIT 100";
    let got = triples(&db, sql);
    assert_eq!(got, triples(&oracle, sql));
    assert_eq!(got.len(), 100);
    assert!(got
        .iter()
        .all(|(a, b, _)| b.is_some_and(|b| b % 2 == 0 && *a < b)));
}

/// Orders of sixty users, forty per user, ten each
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

const GROUPED: &str = "SELECT u.id, COUNT(o.id), SUM(o.amount) FROM users u \
    INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id LIMIT 5";

/// The index still holds an order under the user it had when the
/// transaction started, while the row itself moved to another user: the
/// grouped join checks the fetched row's key and does not charge the moved
/// order to its old user. The index takes the new key at commit, and the
/// order counts under its new user from then on
#[test]
fn a_grouped_join_does_not_charge_a_moved_order_to_its_old_user() {
    let db = Database::open("memory://join_eq_grouped_moved").unwrap();
    orders_in(&db);
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "UPDATE orders SET user_id = 2, amount = 1000 WHERE id = 1",
        (),
    )
    .unwrap();
    let inside = triples(&db, GROUPED);
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(inside[0], (1, Some(39), 390), "not under the old user");
    let after = triples(&db, GROUPED);
    assert_eq!(after[0], (1, Some(39), 390));
    assert_eq!(after[1], (2, Some(41), 1400), "under the new user");
}

/// EXPLAIN names the strategy the join runs with: a join the executor
/// bounds probes the table's index, an unbounded one on a persistent table
/// does not
#[test]
fn explain_reports_the_index_join_only_for_a_bounded_join() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    users_in(&db, 600);
    let plan = |sql: &str| -> String {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let unbounded = "SELECT u1.id, u2.id FROM users u1 INNER JOIN users u2 ON u1.age = u2.age";
    assert!(
        !plan(unbounded).contains("Index Nested Loop"),
        "no limit, no probe: {}",
        plan(unbounded)
    );
    let bounded =
        "SELECT u1.id, u2.id FROM users u1 INNER JOIN users u2 ON u1.age = u2.age LIMIT 5";
    assert!(
        plan(bounded).contains("Index Nested Loop"),
        "{}",
        plan(bounded)
    );
    let ordered = "SELECT u1.id, u2.id FROM users u1 INNER JOIN users u2 ON u1.age = u2.age ORDER BY u1.id LIMIT 5";
    assert!(
        !plan(ordered).contains("Index Nested Loop"),
        "a limit under ORDER BY is not pushed: {}",
        plan(ordered)
    );
}

/// The grouped index join takes plain aggregates over plain group columns;
/// EXPLAIN names it for that shape and not for an aggregate expression the
/// grouped operator turns away
#[test]
fn explain_reports_the_grouped_index_join_only_for_the_shape_it_takes() {
    let dir = tempfile::tempdir().unwrap();
    let db = file_db(&dir);
    orders_in(&db);
    let plan = |sql: &str| -> String {
        db.query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let plain = "SELECT u.id, SUM(o.amount) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id LIMIT 1";
    assert!(plan(plain).contains("Index Nested Loop"), "{}", plan(plain));
    let expression = "SELECT u.id, SUM(o.amount + 1) FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.id LIMIT 1";
    assert!(
        !plan(expression).contains("Index Nested Loop"),
        "an aggregate expression is not the grouped operator's: {}",
        plan(expression)
    );
    let window = "SELECT u.id, SUM(o.amount) OVER () FROM users u INNER JOIN orders o ON u.id = o.user_id LIMIT 1";
    assert!(
        !plan(window).contains("Index Nested Loop"),
        "a window function is not the index join's: {}",
        plan(window)
    );
}
#[cfg(feature = "test-failpoints")]
mod hot_index {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use stoolap::Database;

    use super::{file_db, people_in, probes, triples, users_in, SELF_JOIN};

    /// The join asks the hot index for every outer key it needs and builds
    /// no hash table
    #[test]
    fn a_limited_join_probes_the_hot_index() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        users_in(&db, 2000);
        let oracle = Database::open("memory://join_eq_oracle_probes").unwrap();
        users_in(&oracle, 2000);

        let probed = Arc::new(AtomicUsize::new(0));
        probes::count(Arc::clone(&probed));
        let got = triples(&db, SELF_JOIN);
        probes::stop();
        assert_eq!(got, triples(&oracle, SELF_JOIN));
        assert!(
            probed.load(Ordering::SeqCst) > 0,
            "the hot index answered the join"
        );
    }

    /// Without a limit the join is not bounded, so it takes the hash path
    #[test]
    fn a_join_without_a_limit_takes_the_hash_path() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        users_in(&db, 600);
        let oracle = Database::open("memory://join_eq_oracle_unbounded").unwrap();
        users_in(&oracle, 600);

        let sql = "SELECT u1.id, u2.id, u1.age FROM users u1 \
            INNER JOIN users u2 ON u1.age = u2.age AND u1.id < u2.id";
        let probed = Arc::new(AtomicUsize::new(0));
        probes::count(Arc::clone(&probed));
        let got = triples(&db, sql);
        probes::stop();
        assert_eq!(got, triples(&oracle, sql));
        assert_eq!(probed.load(Ordering::SeqCst), 0);
    }

    /// A key past the cap is refused before anything is copied, and the
    /// join runs on the hash path from the start
    #[test]
    fn a_key_past_the_cap_sends_the_join_to_the_hash_path() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        users_in(&db, 100);
        let insert = db.prepare("INSERT INTO users VALUES (?, ?, ?)").unwrap();
        for id in 10_001..=11_100 {
            insert.execute((id, "odd", 99)).unwrap();
        }
        people_in(&db, &[99]);

        let sql = "SELECT p.id, u.id, p.age FROM people p \
            INNER JOIN users u ON p.age = u.age LIMIT 50";
        let probed = Arc::new(AtomicUsize::new(0));
        probes::count(Arc::clone(&probed));
        let got = triples(&db, sql);
        probes::stop();
        assert_eq!(probed.load(Ordering::SeqCst), 1, "one refused probe");
        assert_eq!(got.len(), 50);
        assert!(got
            .iter()
            .all(|(p, u, age)| *p == 1 && *age == 99 && u.is_some_and(|u| u > 10_000)));
    }

    /// A first outer chunk that ends on a matched row and short of the limit
    /// continues from that row and keeps probing, instead of restarting on a
    /// snapshot the table cannot answer under
    #[test]
    fn a_join_short_of_its_limit_continues_from_the_row_the_chunk_ended_on() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        users_in(&db, 120);
        let mut ages = vec![200; 99];
        ages.push(25);
        ages.extend(std::iter::repeat_n(26, 100));
        people_in(&db, &ages);

        let sql = "SELECT p.id, u.id, p.age FROM people p \
            INNER JOIN users u ON p.age = u.age LIMIT 5";
        let probed = Arc::new(AtomicUsize::new(0));
        probes::count(Arc::clone(&probed));
        let got = triples(&db, sql);
        probes::stop();
        assert_eq!(
            got,
            vec![
                (100, Some(7), 25),
                (100, Some(67), 25),
                (101, Some(8), 26),
                (101, Some(68), 26),
                (102, Some(8), 26),
            ]
        );
        assert!(
            probed.load(Ordering::SeqCst) > 100,
            "the second chunk was probed too"
        );
    }

    /// A seal that lands after the probe was admitted and before it reads
    /// the index would leave the moved rows out; the probe notices and the
    /// join answers from the hash path
    #[test]
    fn a_seal_inside_the_probe_sends_the_join_to_the_hash_path() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        users_in(&db, 2000);
        people_in(&db, &[20]);

        let other = db.clone();
        stoolap::test_failpoints::after_join_probe_admitted(move || {
            other.execute("PRAGMA CHECKPOINT", ()).unwrap();
        });
        let sql = "SELECT p.id, u.id, p.age FROM people p \
            INNER JOIN users u ON p.age = u.age LIMIT 10";
        let got = triples(&db, sql);
        assert_eq!(got.len(), 10, "the sealed rows are still joined");
        assert!(got
            .iter()
            .all(|(_, u, age)| *age == 20 && u.is_some_and(|u| u % 60 == 2)));

        // The rows are in a volume now, so no probe is admitted
        let probed = Arc::new(AtomicUsize::new(0));
        probes::count(Arc::clone(&probed));
        assert_eq!(triples(&db, sql), got);
        probes::stop();
        assert_eq!(probed.load(Ordering::SeqCst), 0);
    }

    /// A truncate inside the second probe: the pairs of the first outer row
    /// are not an answer any more, and the join answers from the hash path,
    /// which finds the table empty
    #[test]
    fn a_truncate_inside_a_later_probe_drops_the_rows_joined_so_far() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        users_in(&db, 2000);
        people_in(&db, &[20, 21]);

        let other = db.clone();
        stoolap::test_failpoints::after_join_probe_admitted(move || {
            stoolap::test_failpoints::after_join_probe_admitted(move || {
                other.execute("TRUNCATE TABLE users", ()).unwrap();
            });
        });
        let sql = "SELECT p.id, u.id, p.age FROM people p \
            INNER JOIN users u ON p.age = u.age LIMIT 1000";
        assert!(
            triples(&db, sql).is_empty(),
            "a join torn by a truncate is not an answer"
        );
    }

    /// A commit inside the probe moves the index epoch: the probe is not
    /// trusted, and the join answers from the hash path instead of probing on
    #[test]
    fn a_commit_inside_the_probe_sends_the_join_to_the_hash_path() {
        let dir = tempfile::tempdir().unwrap();
        let db = file_db(&dir);
        users_in(&db, 2000);
        people_in(&db, &[20, 21]);

        let probed = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&probed);
        let other = db.clone();
        stoolap::test_failpoints::after_join_probe_admitted(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            other
                .execute("INSERT INTO users VALUES (5000, 'even', 20)", ())
                .unwrap();
            probes::count(counter);
        });
        let sql = "SELECT p.id, u.id, p.age FROM people p \
            INNER JOIN users u ON p.age = u.age LIMIT 1000";
        let got = triples(&db, sql);
        probes::stop();
        assert_eq!(
            probed.load(Ordering::SeqCst),
            1,
            "no probe after the commit"
        );
        assert_eq!(got.len(), 69, "both keys, with the committed row");
        assert!(got.contains(&(1, Some(5000), 20)));
    }
}
