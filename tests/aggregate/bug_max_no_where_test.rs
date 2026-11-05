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

//! Debug tests for Bug 2: MAX(time) without WHERE returns wrong result.
//!
//! Hypothesis: The no-WHERE MAX path (index fast path or deferred aggregation)
//! returns a different value than the standard scan path. This could happen if:
//! (a) The BTree index doesn't include all rows, or
//! (b) The deferred aggregation path misses rows, or
//! (c) Timestamps stored in different formats compare incorrectly.

use stoolap::Database;

/// Helper to get a single value from a query as string
fn query_one(db: &Database, sql: &str) -> String {
    let mut rows = db.query(sql, ()).unwrap();
    if let Some(Ok(row)) = rows.next() {
        row.get::<String>(0).unwrap_or_default()
    } else {
        "NO ROWS".to_string()
    }
}

#[test]
fn test_max_timestamp_basic() {
    // Simple case: all timestamps inserted the same way
    let db = Database::open("memory://max_ts_basic").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, '2026-03-10T10:00:00Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, '2026-03-11T10:00:00Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, '2026-03-13T10:00:00Z')", ())
        .unwrap();

    // All three paths should agree
    let max_no_where = query_one(&db, "SELECT MAX(ts) FROM t");
    let max_with_where = query_one(&db, "SELECT MAX(ts) FROM t WHERE id > 0");
    let order_limit = query_one(&db, "SELECT ts FROM t ORDER BY ts DESC LIMIT 1");

    eprintln!("MAX no WHERE:   {}", max_no_where);
    eprintln!("MAX with WHERE: {}", max_with_where);
    eprintln!("ORDER BY LIMIT: {}", order_limit);

    assert_eq!(
        max_no_where, max_with_where,
        "MAX without WHERE should match MAX with WHERE"
    );
}

#[test]
fn test_max_timestamp_multiple_batches() {
    // Insert in separate batches to potentially create multiple arena segments
    let db = Database::open("memory://max_ts_batches").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();

    // Batch 1: older timestamps
    for i in 1..=100 {
        db.execute(
            &format!(
                "INSERT INTO t VALUES ({}, '2026-03-10T{:02}:{:02}:00Z')",
                i,
                i / 60,
                i % 60
            ),
            (),
        )
        .unwrap();
    }

    // Batch 2: newer timestamps
    for i in 101..=200 {
        db.execute(
            &format!(
                "INSERT INTO t VALUES ({}, '2026-03-13T{:02}:{:02}:00Z')",
                i,
                (i - 100) / 60,
                (i - 100) % 60
            ),
            (),
        )
        .unwrap();
    }

    let max_no_where = query_one(&db, "SELECT MAX(ts) FROM t");
    let max_with_where = query_one(&db, "SELECT MAX(ts) FROM t WHERE id > 0");
    let order_limit = query_one(&db, "SELECT ts FROM t ORDER BY ts DESC LIMIT 1");

    eprintln!("MAX no WHERE:   {}", max_no_where);
    eprintln!("MAX with WHERE: {}", max_with_where);
    eprintln!("ORDER BY LIMIT: {}", order_limit);

    assert_eq!(
        max_no_where, max_with_where,
        "MAX without WHERE should match MAX with WHERE across batches"
    );
}

#[test]
fn test_max_timestamp_with_updates() {
    // Updates could cause the max in the index to diverge from the arena
    let db = Database::open("memory://max_ts_updates").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();

    db.execute("INSERT INTO t VALUES (1, '2026-03-10T10:00:00Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, '2026-03-11T10:00:00Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, '2026-03-12T10:00:00Z')", ())
        .unwrap();

    // Update row 3 to have the latest timestamp
    db.execute("UPDATE t SET ts = '2026-03-13T10:00:00Z' WHERE id = 3", ())
        .unwrap();

    let max_no_where = query_one(&db, "SELECT MAX(ts) FROM t");
    let max_with_where = query_one(&db, "SELECT MAX(ts) FROM t WHERE id > 0");
    let order_limit = query_one(&db, "SELECT ts FROM t ORDER BY ts DESC LIMIT 1");

    eprintln!("MAX no WHERE:   {}", max_no_where);
    eprintln!("MAX with WHERE: {}", max_with_where);
    eprintln!("ORDER BY LIMIT: {}", order_limit);

    assert_eq!(
        max_no_where, max_with_where,
        "MAX without WHERE should reflect updated timestamp"
    );
}

#[test]
fn test_max_timestamp_with_deletes() {
    // Deleting the current max row should change the result
    let db = Database::open("memory://max_ts_deletes").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();

    db.execute("INSERT INTO t VALUES (1, '2026-03-10T10:00:00Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, '2026-03-13T10:00:00Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, '2026-03-11T10:00:00Z')", ())
        .unwrap();

    // Delete the row with the max timestamp
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();

    let max_no_where = query_one(&db, "SELECT MAX(ts) FROM t");
    let max_with_where = query_one(&db, "SELECT MAX(ts) FROM t WHERE id > 0");
    let order_limit = query_one(&db, "SELECT ts FROM t ORDER BY ts DESC LIMIT 1");

    eprintln!("MAX no WHERE:   {}", max_no_where);
    eprintln!("MAX with WHERE: {}", max_with_where);
    eprintln!("ORDER BY LIMIT: {}", order_limit);

    assert_eq!(
        max_no_where, max_with_where,
        "MAX without WHERE should exclude deleted rows"
    );
}

#[test]
fn test_max_timestamp_mixed_formats() {
    // Different timestamp format strings that represent the same or different times
    let db = Database::open("memory://max_ts_formats").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();

    // Various timestamp format representations
    db.execute("INSERT INTO t VALUES (1, '2026-03-10T11:31:00.000Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, '2026-03-13T13:14:00+00:00')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (3, '2026-03-12 08:00:00')", ())
        .unwrap();

    let max_no_where = query_one(&db, "SELECT MAX(ts) FROM t");
    let max_with_where = query_one(&db, "SELECT MAX(ts) FROM t WHERE id > 0");
    let order_limit = query_one(&db, "SELECT ts FROM t ORDER BY ts DESC LIMIT 1");

    eprintln!("MAX no WHERE:   {}", max_no_where);
    eprintln!("MAX with WHERE: {}", max_with_where);
    eprintln!("ORDER BY LIMIT: {}", order_limit);

    // The max should be 2026-03-13 regardless of input format
    assert_eq!(
        max_no_where, max_with_where,
        "MAX without WHERE should handle mixed timestamp formats correctly"
    );
}

#[test]
fn test_max_timestamp_large_table_with_index() {
    // Simulate the user's scenario: large table with BTREE index on timestamp column
    let db = Database::open("memory://max_ts_large_idx").unwrap();
    db.execute(
        "CREATE TABLE candlesticks (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            time TIMESTAMP,
            close FLOAT
        )",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_time ON candlesticks (time)", ())
        .unwrap();

    // Insert rows spanning different days (simulating market data)
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let mut id = 1;

    // Day 1: March 10 (oldest)
    for symbol in &symbols {
        for hour in 0..24 {
            db.execute(
                &format!(
                    "INSERT INTO candlesticks VALUES ({}, '{}', '2026-03-10T{:02}:00:00Z', 100.0)",
                    id, symbol, hour
                ),
                (),
            )
            .unwrap();
            id += 1;
        }
    }

    // Day 2: March 13 (newest) - only for some symbols
    for hour in 0..14 {
        db.execute(
            &format!(
                "INSERT INTO candlesticks VALUES ({}, 'BTCUSDT', '2026-03-13T{:02}:14:00Z', 105.0)",
                id, hour
            ),
            (),
        )
        .unwrap();
        id += 1;
    }

    let max_no_where = query_one(&db, "SELECT MAX(time) FROM candlesticks");
    let max_with_symbol = query_one(
        &db,
        "SELECT MAX(time) FROM candlesticks WHERE symbol = 'BTCUSDT'",
    );
    let order_limit = query_one(
        &db,
        "SELECT time FROM candlesticks ORDER BY time DESC LIMIT 1",
    );

    eprintln!("MAX no WHERE:      {}", max_no_where);
    eprintln!("MAX WHERE symbol:  {}", max_with_symbol);
    eprintln!("ORDER BY LIMIT:    {}", order_limit);

    // All should return 2026-03-13T13:14:00Z
    assert_eq!(
        max_no_where, max_with_symbol,
        "MAX without WHERE should match MAX with WHERE (same max across all symbols)"
    );
    assert_eq!(
        max_no_where, order_limit,
        "MAX without WHERE should match ORDER BY DESC LIMIT 1"
    );
}

#[test]
fn test_max_timestamp_after_delete_and_reinsert() {
    // Delete max row, insert a new higher max — tests index + arena consistency
    let db = Database::open("memory://max_ts_reinsert").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_ts ON t (ts)", ()).unwrap();

    db.execute("INSERT INTO t VALUES (1, '2026-03-10T10:00:00Z')", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (2, '2026-03-11T10:00:00Z')", ())
        .unwrap();

    // Check initial max
    let max1 = query_one(&db, "SELECT MAX(ts) FROM t");
    eprintln!("Initial MAX: {}", max1);

    // Delete the max row
    db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();

    let max2 = query_one(&db, "SELECT MAX(ts) FROM t");
    let max2_with_where = query_one(&db, "SELECT MAX(ts) FROM t WHERE id > 0");
    eprintln!("After delete MAX no WHERE:   {}", max2);
    eprintln!("After delete MAX with WHERE: {}", max2_with_where);
    assert_eq!(
        max2, max2_with_where,
        "After delete: MAX without WHERE should match MAX with WHERE"
    );

    // Insert a new row with an even later timestamp
    db.execute("INSERT INTO t VALUES (3, '2026-03-13T10:00:00Z')", ())
        .unwrap();

    let max3 = query_one(&db, "SELECT MAX(ts) FROM t");
    let max3_with_where = query_one(&db, "SELECT MAX(ts) FROM t WHERE id > 0");
    let max3_order = query_one(&db, "SELECT ts FROM t ORDER BY ts DESC LIMIT 1");
    eprintln!("After reinsert MAX no WHERE:   {}", max3);
    eprintln!("After reinsert MAX with WHERE: {}", max3_with_where);
    eprintln!("After reinsert ORDER BY:       {}", max3_order);
    assert_eq!(
        max3, max3_with_where,
        "After reinsert: MAX without WHERE should match MAX with WHERE"
    );
    assert_eq!(
        max3, max3_order,
        "After reinsert: MAX should match ORDER BY DESC LIMIT 1"
    );
}

/// Simulate production scenario: multiple DB open/close cycles with data
/// spanning snapshot boundaries + WAL entries from different sessions.
#[test]
fn test_max_timestamp_across_persistence_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/max_persist", dir.path().display());

    // Session 1: Create table, insert backfill data (older timestamps)
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE candlesticks (
                id INTEGER PRIMARY KEY,
                exchange TEXT,
                symbol TEXT,
                time TIMESTAMP,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume FLOAT
            )",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX idx_time ON candlesticks (time)", ())
            .unwrap();

        // Backfill: insert 100 rows with March 10 timestamps
        for i in 1..=100 {
            db.execute(
                &format!(
                    "INSERT INTO candlesticks VALUES ({}, 'binance', 'BTCUSDT', '2026-03-10T{:02}:{:02}:00Z', 100.0, 101.0, 99.0, 100.5, 1000.0)",
                    i, i / 60, i % 60
                ),
                (),
            )
            .unwrap();
        }

        let max1 = query_one(&db, "SELECT MAX(time) FROM candlesticks");
        eprintln!("Session 1 MAX: {}", max1);
        db.close().unwrap();
    }

    // Session 2: Reopen, insert newer data (March 13)
    {
        let db = Database::open(&dsn).unwrap();

        // Verify recovered data
        let count = query_one(&db, "SELECT COUNT(*) FROM candlesticks");
        eprintln!("Session 2 recovered rows: {}", count);

        // Insert newer timestamps
        for i in 101..=150 {
            db.execute(
                &format!(
                    "INSERT INTO candlesticks VALUES ({}, 'binance', 'BTCUSDT', '2026-03-13T{:02}:{:02}:00Z', 105.0, 106.0, 104.0, 105.5, 2000.0)",
                    i, (i - 100) / 60, (i - 100) % 60
                ),
                (),
            )
            .unwrap();
        }

        let max_no_where = query_one(&db, "SELECT MAX(time) FROM candlesticks");
        let max_with_where = query_one(
            &db,
            "SELECT MAX(time) FROM candlesticks WHERE symbol = 'BTCUSDT'",
        );
        let order_limit = query_one(
            &db,
            "SELECT time FROM candlesticks ORDER BY time DESC LIMIT 1",
        );

        eprintln!("Session 2 MAX no WHERE:   {}", max_no_where);
        eprintln!("Session 2 MAX with WHERE: {}", max_with_where);
        eprintln!("Session 2 ORDER BY LIMIT: {}", order_limit);

        assert_eq!(
            max_no_where, max_with_where,
            "Session 2: MAX without WHERE should match MAX with WHERE"
        );
        assert_eq!(
            max_no_where, order_limit,
            "Session 2: MAX should match ORDER BY DESC LIMIT 1"
        );

        db.close().unwrap();
    }

    // Session 3: Reopen again and verify cross-snapshot MAX
    {
        let db = Database::open(&dsn).unwrap();

        let count = query_one(&db, "SELECT COUNT(*) FROM candlesticks");
        eprintln!("Session 3 recovered rows: {}", count);

        let max_no_where = query_one(&db, "SELECT MAX(time) FROM candlesticks");
        let max_with_where = query_one(
            &db,
            "SELECT MAX(time) FROM candlesticks WHERE symbol = 'BTCUSDT'",
        );
        let order_limit = query_one(
            &db,
            "SELECT time FROM candlesticks ORDER BY time DESC LIMIT 1",
        );

        eprintln!("Session 3 MAX no WHERE:   {}", max_no_where);
        eprintln!("Session 3 MAX with WHERE: {}", max_with_where);
        eprintln!("Session 3 ORDER BY LIMIT: {}", order_limit);

        assert_eq!(
            max_no_where, max_with_where,
            "Session 3: MAX without WHERE should match after recovery"
        );
        assert_eq!(
            max_no_where, order_limit,
            "Session 3: MAX should match ORDER BY after recovery"
        );

        db.close().unwrap();
    }
}

/// Simulate ON CONFLICT updates across persistence boundaries.
/// The production workload uses ON CONFLICT to update existing candlestick rows.
#[test]
fn test_max_timestamp_on_conflict_across_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/max_conflict", dir.path().display());

    // Session 1: Create table with unique constraint, insert initial data
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE candlesticks (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                time TIMESTAMP,
                close FLOAT,
                UNIQUE(symbol, time)
            )",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX idx_time ON candlesticks (time)", ())
            .unwrap();

        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO candlesticks VALUES ({}, 'BTCUSDT', '2026-03-10T{:02}:{:02}:00Z', 100.0)",
                    i, i / 60, i % 60
                ),
                (),
            )
            .unwrap();
        }

        db.close().unwrap();
    }

    // Session 2: Reopen, do ON CONFLICT updates on existing rows + insert newer data
    {
        let db = Database::open(&dsn).unwrap();

        // ON CONFLICT update on existing rows (updates close price, same symbol+time)
        // Use same IDs as session 1 to trigger ON CONFLICT on PK
        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO candlesticks VALUES ({}, 'BTCUSDT', '2026-03-10T{:02}:{:02}:00Z', 105.0)
                     ON CONFLICT (id) DO UPDATE SET close = 105.0",
                    i, i / 60, i % 60
                ),
                (),
            )
            .unwrap();
        }

        // Insert newer data with later timestamps (new IDs)
        for i in 51..=80 {
            db.execute(
                &format!(
                    "INSERT INTO candlesticks VALUES ({}, 'BTCUSDT', '2026-03-13T{:02}:{:02}:00Z', 110.0)",
                    i, (i - 50) / 60, (i - 50) % 60
                ),
                (),
            )
            .unwrap();
        }

        let max_no_where = query_one(&db, "SELECT MAX(time) FROM candlesticks");
        let max_with_where = query_one(
            &db,
            "SELECT MAX(time) FROM candlesticks WHERE symbol = 'BTCUSDT'",
        );
        let order_limit = query_one(
            &db,
            "SELECT time FROM candlesticks ORDER BY time DESC LIMIT 1",
        );

        eprintln!("ON CONFLICT session MAX no WHERE:   {}", max_no_where);
        eprintln!("ON CONFLICT session MAX with WHERE: {}", max_with_where);
        eprintln!("ON CONFLICT session ORDER BY LIMIT: {}", order_limit);

        assert_eq!(
            max_no_where, max_with_where,
            "After ON CONFLICT: MAX without WHERE should match MAX with WHERE"
        );
        assert_eq!(
            max_no_where, order_limit,
            "After ON CONFLICT: MAX should match ORDER BY DESC LIMIT 1"
        );

        db.close().unwrap();
    }

    // Session 3: Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();

        let max_no_where = query_one(&db, "SELECT MAX(time) FROM candlesticks");
        let max_with_where = query_one(
            &db,
            "SELECT MAX(time) FROM candlesticks WHERE symbol = 'BTCUSDT'",
        );
        let order_limit = query_one(
            &db,
            "SELECT time FROM candlesticks ORDER BY time DESC LIMIT 1",
        );

        eprintln!("Post-recovery MAX no WHERE:   {}", max_no_where);
        eprintln!("Post-recovery MAX with WHERE: {}", max_with_where);
        eprintln!("Post-recovery ORDER BY LIMIT: {}", order_limit);

        assert_eq!(
            max_no_where, max_with_where,
            "Post-recovery: MAX without WHERE should match"
        );
        assert_eq!(
            max_no_where, order_limit,
            "Post-recovery: MAX should match ORDER BY"
        );

        db.close().unwrap();
    }
}

/// Regression test: visibility cache in version_store aggregation functions
/// initialized last_txn_id to -1, which collides with RECOVERY_TRANSACTION_ID (-1).
/// This caused all snapshot-recovered rows at the start of the arena to be treated
/// as invisible, producing wrong MIN/MAX/SUM results when no WHERE clause is present.
#[test]
fn test_max_min_recovery_visibility_cache() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/max_vis", dir.path().display());

    // Session 1: Insert data and close (creates snapshot)
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, ts TIMESTAMP, val FLOAT)",
            (),
        )
        .unwrap();

        // Insert 100 rows: timestamps from 2026-03-01 to 2026-03-04
        for i in 1..=100 {
            db.execute(
                &format!(
                    "INSERT INTO t VALUES ({}, '2026-03-{:02}T{:02}:{:02}:00Z', {})",
                    i,
                    1 + (i - 1) / 25, // days 1-4
                    (i * 3) % 24,
                    i % 60,
                    i as f64 * 1.5
                ),
                (),
            )
            .unwrap();
        }

        let max_before = query_one(&db, "SELECT MAX(ts) FROM t");
        let min_before = query_one(&db, "SELECT MIN(ts) FROM t");
        eprintln!("Before close: MAX={} MIN={}", max_before, min_before);

        db.close().unwrap();
    }

    // Session 2: Reopen (snapshot + WAL recovery), then check.
    // After recovery, all snapshot rows have txn_id = RECOVERY_TRANSACTION_ID (-1).
    // The visibility cache bug would make these invisible.
    {
        let db = Database::open(&dsn).unwrap();

        let count = query_one(&db, "SELECT COUNT(*) FROM t");
        eprintln!("After recovery: COUNT={}", count);

        let max_pushdown = query_one(&db, "SELECT MAX(ts) FROM t");
        let max_scan = query_one(&db, "SELECT MAX(ts) FROM t WHERE ts IS NOT NULL");
        let min_pushdown = query_one(&db, "SELECT MIN(ts) FROM t");
        let min_scan = query_one(&db, "SELECT MIN(ts) FROM t WHERE ts IS NOT NULL");
        let sum_pushdown = query_one(&db, "SELECT SUM(val) FROM t");
        let sum_scan = query_one(&db, "SELECT SUM(val) FROM t WHERE val IS NOT NULL");

        eprintln!("MAX pushdown={} scan={}", max_pushdown, max_scan);
        eprintln!("MIN pushdown={} scan={}", min_pushdown, min_scan);
        eprintln!("SUM pushdown={} scan={}", sum_pushdown, sum_scan);

        assert_eq!(
            max_pushdown, max_scan,
            "MAX mismatch after recovery: pushdown={} scan={}",
            max_pushdown, max_scan
        );
        assert_eq!(
            min_pushdown, min_scan,
            "MIN mismatch after recovery: pushdown={} scan={}",
            min_pushdown, min_scan
        );
        assert_eq!(
            sum_pushdown, sum_scan,
            "SUM mismatch after recovery: pushdown={} scan={}",
            sum_pushdown, sum_scan
        );

        db.close().unwrap();
    }
}
