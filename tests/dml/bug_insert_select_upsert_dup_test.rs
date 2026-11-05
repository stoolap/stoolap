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

//! Reproduce reported bug: INSERT INTO ... SELECT ... ON CONFLICT silently inserts
//! duplicates instead of upserting when run a second time with overlapping data.

use stoolap::Database;

#[test]
fn test_insert_select_on_conflict_second_run_duplicates() {
    let db = Database::open("memory://insert_select_dup").unwrap();

    // Target table with composite UNIQUE
    db.execute(
        "CREATE TABLE ohlcv (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            time TIMESTAMP NOT NULL,
            open FLOAT NOT NULL,
            close FLOAT NOT NULL,
            UNIQUE(exchange, symbol, time)
        )",
        (),
    )
    .unwrap();

    // Source table
    db.execute(
        "CREATE TABLE raw_trades (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            time TIMESTAMP NOT NULL,
            price FLOAT NOT NULL
        )",
        (),
    )
    .unwrap();

    // Insert raw trades
    db.execute(
        "INSERT INTO raw_trades (exchange, symbol, time, price) VALUES
         ('binance', 'BTC/USD', '2024-01-01 00:00:00', 42000.0),
         ('binance', 'BTC/USD', '2024-01-01 00:00:00', 42100.0),
         ('binance', 'BTC/USD', '2024-01-01 01:00:00', 43000.0),
         ('binance', 'ETH/USD', '2024-01-01 00:00:00', 2200.0)",
        (),
    )
    .unwrap();

    // First run: INSERT INTO ... SELECT ... ON CONFLICT DO UPDATE
    db.execute(
        "INSERT INTO ohlcv (exchange, symbol, time, open, close)
         SELECT exchange, symbol, time, MIN(price), MAX(price)
         FROM raw_trades
         GROUP BY exchange, symbol, time
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET
             open = EXCLUDED.open, close = EXCLUDED.close",
        (),
    )
    .unwrap();

    // Should have 3 rows after first run
    let count: i64 = db
        .query("SELECT COUNT(*) FROM ohlcv", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 3, "First run should insert 3 rows");

    // Second run with same data — should UPDATE, not insert duplicates
    db.execute(
        "INSERT INTO ohlcv (exchange, symbol, time, open, close)
         SELECT exchange, symbol, time, MIN(price), MAX(price)
         FROM raw_trades
         GROUP BY exchange, symbol, time
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET
             open = EXCLUDED.open, close = EXCLUDED.close",
        (),
    )
    .unwrap();

    // Should STILL have 3 rows, not 6
    let count: i64 = db
        .query("SELECT COUNT(*) FROM ohlcv", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(
        count, 3,
        "Second run should upsert, not duplicate. Got {} rows",
        count
    );
}

#[test]
fn test_insert_select_on_conflict_no_pk_second_run() {
    let db = Database::open("memory://insert_select_dup_nopk").unwrap();

    // Target table WITHOUT PK, only composite UNIQUE
    db.execute(
        "CREATE TABLE metrics (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            time TIMESTAMP NOT NULL,
            value FLOAT NOT NULL,
            UNIQUE(exchange, symbol, time)
        )",
        (),
    )
    .unwrap();

    // Source table
    db.execute(
        "CREATE TABLE raw_data (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            time TIMESTAMP NOT NULL,
            value FLOAT NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO raw_data VALUES
         ('binance', 'BTC', '2024-01-01 00:00:00', 100.0),
         ('binance', 'BTC', '2024-01-01 01:00:00', 200.0),
         ('binance', 'ETH', '2024-01-01 00:00:00', 50.0)",
        (),
    )
    .unwrap();

    // First run
    db.execute(
        "INSERT INTO metrics (exchange, symbol, time, value)
         SELECT exchange, symbol, time, value FROM raw_data
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET value = EXCLUDED.value",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT COUNT(*) FROM metrics", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 3);

    // Second run — should upsert, not duplicate
    db.execute(
        "INSERT INTO metrics (exchange, symbol, time, value)
         SELECT exchange, symbol, time, value FROM raw_data
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET value = EXCLUDED.value",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT COUNT(*) FROM metrics", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(
        count, 3,
        "Second run should upsert, not duplicate. Got {} rows",
        count
    );
}

#[test]
fn test_insert_select_do_nothing_second_run() {
    let db = Database::open("memory://insert_select_dup_donothing").unwrap();

    db.execute(
        "CREATE TABLE kv (
            k TEXT NOT NULL UNIQUE,
            v TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute("CREATE TABLE source (k TEXT NOT NULL, v TEXT NOT NULL)", ())
        .unwrap();

    db.execute("INSERT INTO source VALUES ('a', 'v1'), ('b', 'v2')", ())
        .unwrap();

    // First run
    db.execute(
        "INSERT INTO kv SELECT k, v FROM source ON CONFLICT (k) DO NOTHING",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT COUNT(*) FROM kv", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, 2);

    // Second run — should skip, not duplicate
    db.execute(
        "INSERT INTO kv SELECT k, v FROM source ON CONFLICT (k) DO NOTHING",
        (),
    )
    .unwrap();

    let count: i64 = db
        .query("SELECT COUNT(*) FROM kv", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(
        count, 2,
        "Second run should skip duplicates. Got {} rows",
        count
    );
}
