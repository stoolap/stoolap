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

//! Regression test: ON DUPLICATE KEY UPDATE with multi-column UNIQUE constraints.
//! Previously failed with "unique constraint failed" instead of triggering UPDATE.

use stoolap::Database;

#[test]
fn test_on_duplicate_key_composite_unique_two_columns() {
    let db = Database::open("memory://composite_upsert_2col").unwrap();

    db.execute(
        "CREATE TABLE kv (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            ns TEXT NOT NULL,
            k TEXT NOT NULL,
            val TEXT,
            UNIQUE(ns, k)
        )",
        (),
    )
    .unwrap();

    // Initial insert
    db.execute(
        "INSERT INTO kv (ns, k, val) VALUES ('app', 'theme', 'dark')
         ON DUPLICATE KEY UPDATE val = 'dark'",
        (),
    )
    .unwrap();

    // Upsert same (ns, key) — should UPDATE, not fail
    db.execute(
        "INSERT INTO kv (ns, k, val) VALUES ('app', 'theme', 'light')
         ON DUPLICATE KEY UPDATE val = 'light'",
        (),
    )
    .unwrap();

    let rows = db
        .query("SELECT val FROM kv WHERE ns = 'app' AND k = 'theme'", ())
        .unwrap();
    let results: Vec<String> = rows
        .into_iter()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();
    assert_eq!(results, vec!["light"]);

    // Should still be 1 row, not 2
    let rows = db.query("SELECT COUNT(*) FROM kv", ()).unwrap();
    let count: i64 = rows.into_iter().next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_on_duplicate_key_composite_unique_three_columns() {
    let db = Database::open("memory://composite_upsert_3col").unwrap();

    db.execute(
        "CREATE TABLE ohlcv (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            ts TIMESTAMP NOT NULL,
            price FLOAT NOT NULL,
            UNIQUE(exchange, symbol, ts)
        )",
        (),
    )
    .unwrap();

    // Initial insert
    db.execute(
        "INSERT INTO ohlcv (exchange, symbol, ts, price)
         VALUES ('binance', 'BTC/USD', '2024-01-15 10:00:00', 42000.0)
         ON DUPLICATE KEY UPDATE price = 42000.0",
        (),
    )
    .unwrap();

    // Upsert same (exchange, symbol, ts) — should UPDATE price
    db.execute(
        "INSERT INTO ohlcv (exchange, symbol, ts, price)
         VALUES ('binance', 'BTC/USD', '2024-01-15 10:00:00', 43500.0)
         ON DUPLICATE KEY UPDATE price = 43500.0",
        (),
    )
    .unwrap();

    let rows = db
        .query(
            "SELECT price FROM ohlcv WHERE exchange = 'binance' AND symbol = 'BTC/USD'",
            (),
        )
        .unwrap();
    let results: Vec<f64> = rows
        .into_iter()
        .map(|r| r.unwrap().get::<f64>(0).unwrap())
        .collect();
    assert_eq!(results, vec![43500.0]);

    // Different symbol — should INSERT new row
    db.execute(
        "INSERT INTO ohlcv (exchange, symbol, ts, price)
         VALUES ('binance', 'ETH/USD', '2024-01-15 10:00:00', 2200.0)
         ON DUPLICATE KEY UPDATE price = 2200.0",
        (),
    )
    .unwrap();

    let rows = db.query("SELECT COUNT(*) FROM ohlcv", ()).unwrap();
    let count: i64 = rows.into_iter().next().unwrap().unwrap().get(0).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_on_duplicate_key_composite_unique_with_params() {
    let db = Database::open("memory://composite_upsert_params").unwrap();

    db.execute(
        "CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            host TEXT NOT NULL,
            metric TEXT NOT NULL,
            value FLOAT NOT NULL,
            UNIQUE(host, metric)
        )",
        (),
    )
    .unwrap();

    let stmt = db
        .prepare(
            "INSERT INTO metrics (host, metric, value)
             VALUES ($1, $2, $3)
             ON DUPLICATE KEY UPDATE value = $3",
        )
        .unwrap();

    // Insert initial values
    stmt.execute(("server1", "cpu", 45.0)).unwrap();
    stmt.execute(("server1", "mem", 72.0)).unwrap();
    stmt.execute(("server2", "cpu", 30.0)).unwrap();

    // Upsert — should update existing rows
    stmt.execute(("server1", "cpu", 88.0)).unwrap();
    stmt.execute(("server1", "mem", 95.0)).unwrap();

    let rows = db
        .query(
            "SELECT host, metric, value FROM metrics ORDER BY host, metric",
            (),
        )
        .unwrap();
    let results: Vec<(String, String, f64)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<String>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<f64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0], ("server1".to_string(), "cpu".to_string(), 88.0));
    assert_eq!(results[1], ("server1".to_string(), "mem".to_string(), 95.0));
    assert_eq!(results[2], ("server2".to_string(), "cpu".to_string(), 30.0));
}

#[test]
fn test_on_duplicate_key_insert_select_pk() {
    let db = Database::open("memory://insert_select_upsert_pk").unwrap();

    db.execute(
        "CREATE TABLE src (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE dst (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
        (),
    )
    .unwrap();

    // Seed destination
    db.execute("INSERT INTO dst VALUES (1, 'alice', 10)", ())
        .unwrap();
    db.execute("INSERT INTO dst VALUES (2, 'bob', 20)", ())
        .unwrap();

    // Seed source with overlapping + new rows
    db.execute("INSERT INTO src VALUES (2, 'bob_v2', 25)", ())
        .unwrap();
    db.execute("INSERT INTO src VALUES (3, 'charlie', 30)", ())
        .unwrap();

    // INSERT ... SELECT ... ON DUPLICATE KEY UPDATE using EXCLUDED to reference incoming values
    db.execute(
        "INSERT INTO dst (id, name, score)
         SELECT id, name, score FROM src
         ON DUPLICATE KEY UPDATE name = EXCLUDED.name, score = EXCLUDED.score",
        (),
    )
    .unwrap();

    let rows = db
        .query("SELECT id, name, score FROM dst ORDER BY id", ())
        .unwrap();
    let results: Vec<(i64, String, i64)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(results.len(), 3);
    // id=1 untouched
    assert_eq!(results[0], (1, "alice".to_string(), 10));
    // id=2 was duplicate — updated with values from src
    assert_eq!(results[1], (2, "bob_v2".to_string(), 25));
    // id=3 was new — inserted
    assert_eq!(results[2], (3, "charlie".to_string(), 30));
}

#[test]
fn test_on_duplicate_key_insert_select_unique() {
    let db = Database::open("memory://insert_select_upsert_unique").unwrap();

    db.execute(
        "CREATE TABLE staging (host TEXT, metric TEXT, value FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            host TEXT NOT NULL,
            metric TEXT NOT NULL,
            value FLOAT NOT NULL,
            UNIQUE(host, metric)
        )",
        (),
    )
    .unwrap();

    // Seed metrics
    db.execute(
        "INSERT INTO metrics (host, metric, value) VALUES ('s1', 'cpu', 50.0)",
        (),
    )
    .unwrap();

    // Staging has overlap + new
    db.execute("INSERT INTO staging VALUES ('s1', 'cpu', 88.0)", ())
        .unwrap();
    db.execute("INSERT INTO staging VALUES ('s1', 'mem', 72.0)", ())
        .unwrap();

    // Upsert from staging into metrics using EXCLUDED to get incoming values
    db.execute(
        "INSERT INTO metrics (host, metric, value)
         SELECT host, metric, value FROM staging
         ON DUPLICATE KEY UPDATE value = EXCLUDED.value",
        (),
    )
    .unwrap();

    let rows = db
        .query(
            "SELECT host, metric, value FROM metrics ORDER BY metric",
            (),
        )
        .unwrap();
    let results: Vec<(String, String, f64)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<String>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<f64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(results.len(), 2);
    // cpu was duplicate — updated with staging value 88.0
    assert_eq!(results[0], ("s1".to_string(), "cpu".to_string(), 88.0));
    // mem was new — inserted with 72.0
    assert_eq!(results[1], ("s1".to_string(), "mem".to_string(), 72.0));
}

#[test]
fn test_on_duplicate_key_excluded_with_values() {
    let db = Database::open("memory://excluded_values").unwrap();

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO items VALUES (1, 'apple', 1.50)", ())
        .unwrap();

    // EXCLUDED should also work with INSERT ... VALUES
    db.execute(
        "INSERT INTO items VALUES (1, 'green apple', 2.00)
         ON DUPLICATE KEY UPDATE name = EXCLUDED.name, price = EXCLUDED.price",
        (),
    )
    .unwrap();

    let rows = db.query("SELECT id, name, price FROM items", ()).unwrap();
    let results: Vec<(i64, String, f64)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<f64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], (1, "green apple".to_string(), 2.0));
}

// ===== PostgreSQL-style ON CONFLICT tests =====

#[test]
fn test_on_conflict_do_nothing_pk() {
    let db = Database::open("memory://conflict_do_nothing_pk").unwrap();

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO items VALUES (1, 'apple')", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (2, 'banana')", ())
        .unwrap();

    // Duplicate id=1 should be silently skipped
    db.execute(
        "INSERT INTO items VALUES (1, 'cherry') ON CONFLICT DO NOTHING",
        (),
    )
    .unwrap();

    let rows = db
        .query("SELECT id, name FROM items ORDER BY id", ())
        .unwrap();
    let results: Vec<(i64, String)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<String>(1).unwrap())
        })
        .collect();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], (1, "apple".to_string())); // unchanged
    assert_eq!(results[1], (2, "banana".to_string()));
}

#[test]
fn test_on_conflict_do_nothing_with_target() {
    let db = Database::open("memory://conflict_do_nothing_target").unwrap();

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, code TEXT, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE UNIQUE INDEX idx_code ON items(code)", ())
        .unwrap();

    db.execute("INSERT INTO items VALUES (1, 'A', 'apple')", ())
        .unwrap();

    // Conflict on code='A', should be silently skipped
    db.execute(
        "INSERT INTO items VALUES (2, 'A', 'avocado') ON CONFLICT (code) DO NOTHING",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT COUNT(*) FROM items", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_on_conflict_do_update_set() {
    let db = Database::open("memory://conflict_do_update").unwrap();

    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price FLOAT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO items VALUES (1, 'apple', 1.50)", ())
        .unwrap();

    // PostgreSQL-style upsert
    db.execute(
        "INSERT INTO items VALUES (1, 'green apple', 2.00)
         ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, price = EXCLUDED.price",
        (),
    )
    .unwrap();

    let rows = db.query("SELECT id, name, price FROM items", ()).unwrap();
    let results: Vec<(i64, String, f64)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<f64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], (1, "green apple".to_string(), 2.0));
}

#[test]
fn test_on_conflict_do_update_with_expression() {
    let db = Database::open("memory://conflict_do_update_expr").unwrap();

    db.execute(
        "CREATE TABLE counters (id INTEGER PRIMARY KEY, name TEXT, count INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO counters VALUES (1, 'visits', 10)", ())
        .unwrap();

    // Increment count on conflict
    db.execute(
        "INSERT INTO counters VALUES (1, 'visits', 5)
         ON CONFLICT (id) DO UPDATE SET count = count + EXCLUDED.count",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT count FROM counters WHERE id = 1", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 15); // 10 + 5
}

#[test]
fn test_on_conflict_do_nothing_insert_select() {
    let db = Database::open("memory://conflict_nothing_select").unwrap();

    db.execute("CREATE TABLE src (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute("CREATE TABLE dst (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();

    db.execute("INSERT INTO dst VALUES (1, 'existing')", ())
        .unwrap();

    db.execute("INSERT INTO src VALUES (1, 'conflict')", ())
        .unwrap();
    db.execute("INSERT INTO src VALUES (2, 'new')", ()).unwrap();

    // Bulk insert with DO NOTHING
    db.execute(
        "INSERT INTO dst SELECT id, name FROM src ON CONFLICT DO NOTHING",
        (),
    )
    .unwrap();

    let rows = db
        .query("SELECT id, name FROM dst ORDER BY id", ())
        .unwrap();
    let results: Vec<(i64, String)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<String>(1).unwrap())
        })
        .collect();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], (1, "existing".to_string())); // unchanged
    assert_eq!(results[1], (2, "new".to_string())); // inserted
}

#[test]
fn test_on_conflict_do_update_insert_select() {
    let db = Database::open("memory://conflict_update_select").unwrap();

    db.execute(
        "CREATE TABLE src (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE dst (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO dst VALUES (1, 'alice', 10)", ())
        .unwrap();

    db.execute("INSERT INTO src VALUES (1, 'alice_v2', 99)", ())
        .unwrap();
    db.execute("INSERT INTO src VALUES (2, 'bob', 50)", ())
        .unwrap();

    // PostgreSQL-style bulk upsert
    db.execute(
        "INSERT INTO dst (id, name, score)
         SELECT id, name, score FROM src
         ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, score = EXCLUDED.score",
        (),
    )
    .unwrap();

    let rows = db
        .query("SELECT id, name, score FROM dst ORDER BY id", ())
        .unwrap();
    let results: Vec<(i64, String, i64)> = rows
        .into_iter()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<String>(1).unwrap(),
                r.get::<i64>(2).unwrap(),
            )
        })
        .collect();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0], (1, "alice_v2".to_string(), 99));
    assert_eq!(results[1], (2, "bob".to_string(), 50));
}
