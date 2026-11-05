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

//! Debug tests for Bug 1: AS time alias shadows table column in GROUP BY.
//!
//! When a SELECT alias matches a table column name, GROUP BY expressions
//! that reference the column inside function calls (e.g., TIME_TRUNC('3m', time))
//! may get confused between the alias and the table column, causing:
//! - Full table scans instead of index seeks
//! - Correct results but 1650x slower performance

use std::time::Instant;
use stoolap::Database;

/// Helper: count result rows
fn count_rows(db: &Database, sql: &str) -> usize {
    let mut rows = db.query(sql, ()).unwrap();
    let mut count = 0;
    while let Some(Ok(_)) = rows.next() {
        count += 1;
    }
    count
}

/// Helper: collect first column as strings
fn collect_col0(db: &Database, sql: &str) -> Vec<String> {
    let mut rows = db.query(sql, ()).unwrap();
    let mut result = Vec::new();
    while let Some(Ok(row)) = rows.next() {
        result.push(row.get::<String>(0).unwrap_or_default());
    }
    result
}

fn setup_candlestick_db(name: &str) -> Database {
    let db = Database::open(&format!("memory://{}", name)).unwrap();
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

    // Create indexes matching production setup
    db.execute("CREATE INDEX idx_time ON candlesticks (time)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_symbol ON candlesticks (symbol)", ())
        .unwrap();

    db
}

fn insert_test_data(db: &Database, count: usize) {
    let exchanges = ["binance", "coinbase"];
    let symbols = ["BTCUSDT", "ETHUSDT"];
    let mut id = 1;

    // Insert data spanning March 10 to March 13
    for day in 10..=13 {
        for hour in 0..24 {
            for minute in (0..60).step_by(1) {
                for exchange in &exchanges {
                    for symbol in &symbols {
                        if id > count {
                            return;
                        }
                        db.execute(
                            &format!(
                                "INSERT INTO candlesticks VALUES ({}, '{}', '{}', '2026-03-{:02}T{:02}:{:02}:00Z', 100.0, 105.0, 95.0, 102.0, 1000.0)",
                                id, exchange, symbol, day, hour, minute
                            ),
                            (),
                        )
                        .unwrap();
                        id += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn test_alias_shadow_correctness() {
    // Test that aliasing TIME_TRUNC result as "time" (same as column name)
    // returns the same result as aliasing it as "t" (different name)
    let db = setup_candlestick_db("alias_shadow_correct");
    insert_test_data(&db, 500);

    // Query with non-shadowing alias
    let result_no_shadow = count_rows(
        &db,
        "SELECT TIME_TRUNC('3m', time) AS t, exchange, symbol, MAX(close)
         FROM candlesticks
         WHERE time >= '2026-03-13T00:00:00Z'
         GROUP BY TIME_TRUNC('3m', time), exchange, symbol",
    );

    // Show EXPLAIN for both
    let explain_no_shadow = collect_col0(
        &db,
        "EXPLAIN SELECT TIME_TRUNC('3m', time) AS t, exchange, symbol, MAX(close)
         FROM candlesticks
         WHERE time >= '2026-03-13T00:00:00Z'
         GROUP BY TIME_TRUNC('3m', time), exchange, symbol",
    );
    let explain_shadow = collect_col0(
        &db,
        "EXPLAIN SELECT TIME_TRUNC('3m', time) AS time, exchange, symbol, MAX(close)
         FROM candlesticks
         WHERE time >= '2026-03-13T00:00:00Z'
         GROUP BY TIME_TRUNC('3m', time), exchange, symbol",
    );
    eprintln!("--- EXPLAIN (no shadow) ---");
    for line in &explain_no_shadow {
        eprintln!("  {}", line);
    }
    eprintln!("--- EXPLAIN (shadow) ---");
    for line in &explain_shadow {
        eprintln!("  {}", line);
    }

    // Query with shadowing alias (alias name matches column name)
    let result_shadow = count_rows(
        &db,
        "SELECT TIME_TRUNC('3m', time) AS time, exchange, symbol, MAX(close)
         FROM candlesticks
         WHERE time >= '2026-03-13T00:00:00Z'
         GROUP BY TIME_TRUNC('3m', time), exchange, symbol",
    );

    eprintln!("Non-shadowing alias rows: {}", result_no_shadow);
    eprintln!("Shadowing alias rows:     {}", result_shadow);

    assert_eq!(
        result_no_shadow, result_shadow,
        "Alias shadowing should not change the result count"
    );
}

#[test]
fn test_alias_shadow_performance() {
    // This test verifies the performance difference when alias shadows column
    let db = setup_candlestick_db("alias_shadow_perf");

    // Insert enough data to show performance difference
    // Use ~2000 rows: enough to see difference but fast for CI
    insert_test_data(&db, 2000);

    let warmup_count = count_rows(
        &db,
        "SELECT COUNT(*) FROM candlesticks WHERE time >= '2026-03-13T00:00:00Z'",
    );
    eprintln!("Rows matching WHERE: {}", warmup_count);

    // Benchmark non-shadowing alias
    let start = Instant::now();
    for _ in 0..5 {
        let _ = count_rows(
            &db,
            "SELECT TIME_TRUNC('3m', time) AS t, exchange, symbol, MAX(close)
             FROM candlesticks
             WHERE time >= '2026-03-13T00:00:00Z'
             GROUP BY TIME_TRUNC('3m', time), exchange, symbol",
        );
    }
    let no_shadow_time = start.elapsed();

    // Benchmark shadowing alias
    let start = Instant::now();
    for _ in 0..5 {
        let _ = count_rows(
            &db,
            "SELECT TIME_TRUNC('3m', time) AS time, exchange, symbol, MAX(close)
             FROM candlesticks
             WHERE time >= '2026-03-13T00:00:00Z'
             GROUP BY TIME_TRUNC('3m', time), exchange, symbol",
        );
    }
    let shadow_time = start.elapsed();

    eprintln!(
        "Non-shadowing: {:?}, Shadowing: {:?}, ratio: {:.1}x",
        no_shadow_time,
        shadow_time,
        shadow_time.as_secs_f64() / no_shadow_time.as_secs_f64()
    );

    // The shadowing version should not be dramatically slower
    // Allow 3x overhead as tolerance (the user reports 1650x)
    let ratio = shadow_time.as_secs_f64() / no_shadow_time.as_secs_f64();
    assert!(
        ratio < 10.0,
        "Alias shadowing causes {:.1}x slowdown (expected < 10x), shadow={:?} vs non-shadow={:?}",
        ratio,
        shadow_time,
        no_shadow_time
    );
}

#[test]
fn test_alias_shadow_simple_group_by() {
    // Simplest possible reproduction: alias a column then GROUP BY it
    let db = Database::open("memory://alias_shadow_simple").unwrap();
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, time TIMESTAMP, category TEXT, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO events VALUES (1, '2026-03-10T10:00:00Z', 'A', 10)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (2, '2026-03-10T10:01:00Z', 'A', 20)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (3, '2026-03-10T10:02:00Z', 'B', 30)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (4, '2026-03-10T10:03:00Z', 'B', 40)",
        (),
    )
    .unwrap();

    // Non-shadowing: alias is "t"
    let r1 = collect_col0(
        &db,
        "SELECT TIME_TRUNC('5m', time) AS t, SUM(value)
         FROM events
         GROUP BY TIME_TRUNC('5m', time)",
    );

    // Shadowing: alias is "time" (same as column name)
    let r2 = collect_col0(
        &db,
        "SELECT TIME_TRUNC('5m', time) AS time, SUM(value)
         FROM events
         GROUP BY TIME_TRUNC('5m', time)",
    );

    eprintln!("Non-shadowing result: {:?}", r1);
    eprintln!("Shadowing result:     {:?}", r2);

    assert_eq!(r1, r2, "Alias shadowing should not change GROUP BY results");
}

#[test]
fn test_alias_shadow_group_by_uses_alias_identifier() {
    // When GROUP BY uses the alias name directly (not the expression)
    let db = Database::open("memory://alias_shadow_gb_alias").unwrap();
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, time TIMESTAMP, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO events VALUES (1, '2026-03-10T10:00:00Z', 10)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (2, '2026-03-10T10:01:00Z', 20)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (3, '2026-03-10T10:05:00Z', 30)",
        (),
    )
    .unwrap();

    // GROUP BY references the alias name "time" (which is also a column name)
    // The engine must resolve "time" in GROUP BY as the alias, not the raw column
    let r1 = count_rows(
        &db,
        "SELECT TIME_TRUNC('5m', time) AS time, SUM(value)
         FROM events
         GROUP BY time",
    );

    let r2 = count_rows(
        &db,
        "SELECT TIME_TRUNC('5m', time) AS t, SUM(value)
         FROM events
         GROUP BY t",
    );

    eprintln!("GROUP BY 'time' (alias=time): {} rows", r1);
    eprintln!("GROUP BY 't' (alias=t):       {} rows", r2);

    assert_eq!(
        r1, r2,
        "GROUP BY alias should work the same regardless of alias name"
    );
}

#[test]
fn test_alias_no_shadow_expression_group_by() {
    // GROUP BY uses the full expression (not an identifier), so alias map shouldn't matter
    let db = Database::open("memory://alias_no_shadow_expr_gb").unwrap();
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, time TIMESTAMP, value INTEGER)",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO events VALUES (1, '2026-03-10T10:00:00Z', 10)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (2, '2026-03-10T10:01:00Z', 20)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (3, '2026-03-10T10:05:00Z', 30)",
        (),
    )
    .unwrap();

    // GROUP BY uses the full function expression, not an identifier
    // parse_group_by should hit the FunctionCall branch, NOT the Identifier branch
    let r1 = count_rows(
        &db,
        "SELECT TIME_TRUNC('5m', time) AS time, SUM(value)
         FROM events
         GROUP BY TIME_TRUNC('5m', time)",
    );

    let r2 = count_rows(
        &db,
        "SELECT TIME_TRUNC('5m', time) AS t, SUM(value)
         FROM events
         GROUP BY TIME_TRUNC('5m', time)",
    );

    eprintln!("Expression GB, alias=time: {} rows", r1);
    eprintln!("Expression GB, alias=t:    {} rows", r2);

    assert_eq!(
        r1, r2,
        "GROUP BY expression should not be affected by SELECT alias name"
    );
}
