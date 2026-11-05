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

//! Reproduction test for INSERT INTO SELECT ON CONFLICT duplicates
//! when existing data is in cold volumes.

use stoolap::Database;

fn query_i64(db: &Database, sql: &str) -> i64 {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

#[test]
fn test_upsert_cold_composite_unique_no_dupes() {
    let db = Database::open("memory://upsert_cold_dupe").unwrap();

    db.execute(
        "CREATE TABLE source (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            time TIMESTAMP,
            exchange TEXT,
            symbol TEXT,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT
        )",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE target (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            time TIMESTAMP,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT,
            UNIQUE(exchange, symbol, time)
        )",
        (),
    )
    .unwrap();

    // Insert source data
    for i in 0..50 {
        let ts = format!("2024-01-01 00:{:02}:00", i);
        db.execute(
            &format!(
                "INSERT INTO source (time, exchange, symbol, open, high, low, close, volume) VALUES ('{}', 'binance', 'BTCUSDT', {}.0, {}.0, {}.0, {}.0, 100.0)",
                ts, 100 + i, 110 + i, 90 + i, 105 + i
            ),
            (),
        )
        .unwrap();
    }
    for i in 0..50 {
        let ts = format!("2024-01-01 00:{:02}:00", i);
        db.execute(
            &format!(
                "INSERT INTO source (time, exchange, symbol, open, high, low, close, volume) VALUES ('{}', 'binance', 'ETHUSDT', {}.0, {}.0, {}.0, {}.0, 50.0)",
                ts, 50 + i, 60 + i, 40 + i, 55 + i
            ),
            (),
        )
        .unwrap();
    }

    let upsert_btc = "INSERT INTO target (time, exchange, symbol, open, high, low, close, volume)
         SELECT
           TIME_TRUNC('5m', time), exchange, symbol,
           FIRST(open ORDER BY time),
           MAX(high),
           MIN(low),
           LAST(close ORDER BY time),
           SUM(volume)
         FROM source
         WHERE exchange = 'binance' AND symbol = 'BTCUSDT'
         GROUP BY TIME_TRUNC('5m', time), exchange, symbol
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET
           open = EXCLUDED.open, high = EXCLUDED.high,
           low = EXCLUDED.low, close = EXCLUDED.close,
           volume = EXCLUDED.volume";

    let upsert_eth = "INSERT INTO target (time, exchange, symbol, open, high, low, close, volume)
         SELECT
           TIME_TRUNC('5m', time), exchange, symbol,
           FIRST(open ORDER BY time),
           MAX(high),
           MIN(low),
           LAST(close ORDER BY time),
           SUM(volume)
         FROM source
         WHERE exchange = 'binance' AND symbol = 'ETHUSDT'
         GROUP BY TIME_TRUNC('5m', time), exchange, symbol
         ON CONFLICT (exchange, symbol, time) DO UPDATE SET
           open = EXCLUDED.open, high = EXCLUDED.high,
           low = EXCLUDED.low, close = EXCLUDED.close,
           volume = EXCLUDED.volume";

    db.execute(upsert_btc, ()).unwrap();
    db.execute(upsert_eth, ()).unwrap();

    let first_count = query_i64(&db, "SELECT COUNT(*) FROM target");
    eprintln!("After first INSERT INTO SELECT: {} rows", first_count);
    assert!(first_count > 0, "No rows inserted");

    // Seal to cold
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    let after_seal = query_i64(&db, "SELECT COUNT(*) FROM target");
    assert_eq!(first_count, after_seal, "Row count changed after seal");

    // Second run: same data, should upsert (not create dupes)
    db.execute(upsert_btc, ()).unwrap();
    db.execute(upsert_eth, ()).unwrap();

    let second_count = query_i64(&db, "SELECT COUNT(*) FROM target");
    eprintln!(
        "After second INSERT INTO SELECT: {} rows (expected {})",
        second_count, first_count
    );

    // Check for exact duplicates
    let mut dupes = db
        .query(
            "SELECT exchange, symbol, time, COUNT(*) as cnt
             FROM target
             GROUP BY exchange, symbol, time
             HAVING COUNT(*) > 1",
            (),
        )
        .unwrap();
    let mut dupe_count = 0;
    while let Some(Ok(row)) = dupes.next() {
        dupe_count += 1;
        eprintln!(
            "DUPE: exchange={}, symbol={}, time={}, count={}",
            row.get::<String>(0).unwrap_or_default(),
            row.get::<String>(1).unwrap_or_default(),
            row.get::<String>(2).unwrap_or_default(),
            row.get::<i64>(3).unwrap_or_default(),
        );
    }

    assert_eq!(dupe_count, 0, "Found {} duplicate groups!", dupe_count);
    assert_eq!(
        first_count, second_count,
        "Row count changed: {} -> {}",
        first_count, second_count
    );
}

/// Test with disk-backed DB and larger data to reproduce production conditions.
#[test]
fn test_upsert_cold_disk_backed() {
    let dir = tempfile::tempdir().unwrap();
    let path = format!("file://{}/testdb", dir.path().display());
    let db = Database::open(&path).unwrap();

    db.execute(
        "CREATE TABLE candlesticks_t1m (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            time TIMESTAMP NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT
        )",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE candlesticks_t5m (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            time TIMESTAMP NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT,
            UNIQUE(exchange, symbol, time)
        )",
        (),
    )
    .unwrap();

    // Insert 500 rows of 1m data
    for i in 0..500 {
        let hour = i / 60;
        let min = i % 60;
        let ts = format!("2024-01-01 {:02}:{:02}:00", hour, min);
        db.execute(
            &format!(
                "INSERT INTO candlesticks_t1m (time, exchange, symbol, open, high, low, close, volume) \
                 VALUES ('{}', 'binance_futures', 'BTCUSDT', {}.5, {}.9, {}.1, {}.7, 1000.0)",
                ts, 40000 + i, 40000 + i, 40000 + i, 40000 + i
            ),
            (),
        )
        .unwrap();
    }

    let upsert_sql =
        "INSERT INTO candlesticks_t5m (time, exchange, symbol, open, high, low, close, volume)
        SELECT
          TIME_TRUNC('5m', time), exchange, symbol,
          FIRST(open ORDER BY time),
          MAX(high),
          MIN(low),
          LAST(close ORDER BY time),
          SUM(volume)
        FROM candlesticks_t1m
        WHERE exchange = 'binance_futures' AND symbol = 'BTCUSDT'
        GROUP BY TIME_TRUNC('5m', time), exchange, symbol
        ON CONFLICT (exchange, symbol, time) DO UPDATE SET
          open = EXCLUDED.open, high = EXCLUDED.high,
          low = EXCLUDED.low, close = EXCLUDED.close,
          volume = EXCLUDED.volume";

    // First run
    db.execute(upsert_sql, ()).unwrap();
    let first = query_i64(&db, "SELECT COUNT(*) FROM candlesticks_t5m");
    eprintln!("Run 1: {} rows in t5m", first);
    assert!(first > 0);

    // Seal to cold
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // Second run — should all be upserts
    db.execute(upsert_sql, ()).unwrap();
    let second = query_i64(&db, "SELECT COUNT(*) FROM candlesticks_t5m");
    eprintln!("Run 2: {} rows (expected {})", second, first);

    let mut dupes = db.query(
        "SELECT exchange, symbol, time, COUNT(*) as cnt FROM candlesticks_t5m GROUP BY exchange, symbol, time HAVING COUNT(*) > 1",
        (),
    ).unwrap();
    let mut dupe_count = 0;
    while let Some(Ok(row)) = dupes.next() {
        dupe_count += 1;
        eprintln!(
            "DUPE: {}, {}, {}, count={}",
            row.get::<String>(0).unwrap_or_default(),
            row.get::<String>(1).unwrap_or_default(),
            row.get::<String>(2).unwrap_or_default(),
            row.get::<i64>(3).unwrap_or_default()
        );
    }
    assert_eq!(
        dupe_count, 0,
        "Found {} dupe groups after run 2",
        dupe_count
    );
    assert_eq!(first, second, "Count changed: {} -> {}", first, second);

    // Third run with another seal
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute(upsert_sql, ()).unwrap();
    let third = query_i64(&db, "SELECT COUNT(*) FROM candlesticks_t5m");
    eprintln!("Run 3: {} rows (expected {})", third, first);
    assert_eq!(
        first, third,
        "Count changed after run 3: {} -> {}",
        first, third
    );
}

#[test]
fn test_upsert_cold_multiple_rounds() {
    let db = Database::open("memory://upsert_cold_rounds").unwrap();

    db.execute(
        "CREATE TABLE candles (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            time TIMESTAMP NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            close FLOAT,
            UNIQUE(exchange, symbol, time)
        )",
        (),
    )
    .unwrap();

    for i in 0..20 {
        let ts = format!("2024-01-01 {:02}:00:00", i);
        db.execute(
            &format!(
                "INSERT INTO candles (time, exchange, symbol, close) VALUES ('{}', 'binance', 'BTC', {})",
                ts, 100.0 + i as f64
            ),
            (),
        )
        .unwrap();
    }

    let initial = query_i64(&db, "SELECT COUNT(*) FROM candles");
    assert_eq!(initial, 20);

    for round in 0..5 {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        for i in 0..20 {
            let ts = format!("2024-01-01 {:02}:00:00", i);
            db.execute(
                &format!(
                    "INSERT INTO candles (time, exchange, symbol, close) VALUES ('{}', 'binance', 'BTC', {})
                     ON CONFLICT (exchange, symbol, time) DO UPDATE SET close = EXCLUDED.close",
                    ts,
                    200.0 + i as f64 + round as f64
                ),
                (),
            )
            .unwrap();
        }

        let count = query_i64(&db, "SELECT COUNT(*) FROM candles");
        assert_eq!(
            count, 20,
            "Round {}: expected 20 rows, got {} (dupes!)",
            round, count
        );
    }
}
