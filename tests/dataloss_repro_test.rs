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

use stoolap::Database;

fn count(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

#[test]
fn test_insert_checkpoint_insert_more_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/dataloss1", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO t VALUES ($1, $2)").unwrap();
        for i in 0..100_000i64 {
            stmt.execute((i, i)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 100_000);

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        db.execute("BEGIN", ()).unwrap();
        for i in 100_000..120_000i64 {
            stmt.execute((i, i)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 120_000);

        db.close().unwrap();
    }
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 120_000);
        db.close().unwrap();
    }
}

#[test]
fn test_upsert_cold_rows_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/dataloss2", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER, name TEXT)",
            (),
        )
        .unwrap();
        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO t VALUES ($1, $2, $3)").unwrap();
        for i in 0..100_000i64 {
            stmt.execute((i, i, format!("orig_{}", i))).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 100_000);

        // Upsert 10K rows (update cold rows)
        db.execute("BEGIN", ()).unwrap();
        let upsert = db
            .prepare("INSERT INTO t VALUES ($1, $2, $3) ON CONFLICT(id) DO UPDATE SET val = EXCLUDED.val, name = EXCLUDED.name")
            .unwrap();
        for i in 0..10_000i64 {
            upsert
                .execute((i, i + 999, format!("updated_{}", i)))
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 100_000);

        db.close().unwrap();
    }
    {
        let db = Database::open(&dsn).unwrap();
        let c = count(&db, "SELECT COUNT(*) FROM t");
        assert_eq!(c, 100_000, "Expected 100000 after upsert+reopen, got {}", c);

        // Verify updated values
        let val = count(&db, "SELECT val FROM t WHERE id = 0");
        assert_eq!(val, 999, "Expected updated val=999, got {}", val);

        db.close().unwrap();
    }
}

/// GROUP BY on multi-column key must correctly count duplicates.
/// Bug: global GROUP BY HAVING COUNT(*)>1 returns fewer groups than
/// per-symbol GROUP BY on the same data.
#[test]
fn test_group_by_having_count_correctness() {
    let db = Database::open("memory://group_by_bug").unwrap();

    db.execute(
        "CREATE TABLE candles (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            time TIMESTAMP NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open FLOAT
        )",
        (),
    )
    .unwrap();

    // Insert rows where each (exchange, symbol, time) has exactly 2 copies
    // Use enough symbols and times to create hash pressure
    let symbols: Vec<String> = (0..30).map(|i| format!("SYM{:03}USDT", i)).collect();
    let stmt = db
        .prepare(
            "INSERT INTO candles (time, exchange, symbol, open) VALUES ($1, 'binance', $2, $3)",
        )
        .unwrap();

    db.execute("BEGIN", ()).unwrap();
    for sym in &symbols {
        for day in 1..=28 {
            for hour in 0..24 {
                for minute in (0..60).step_by(5) {
                    let ts = format!("2026-02-{:02} {:02}:{:02}:00", day, hour, minute);
                    // Insert same row TWICE (different id, same business key)
                    stmt.execute((ts.as_str(), sym.as_str(), 100.0f64)).unwrap();
                    stmt.execute((ts.as_str(), sym.as_str(), 100.0f64)).unwrap();
                }
            }
        }
    }
    db.execute("COMMIT", ()).unwrap();

    let total = count(&db, "SELECT COUNT(*) FROM candles");
    let expected_pairs = 30 * 28 * 24 * 12; // 30 syms * 28 days * 288 slots
    eprintln!("Total rows: {} (expected {})", total, expected_pairs * 2);
    assert_eq!(total, expected_pairs * 2);

    // Per-symbol GROUP BY
    let per_sym = count(
        &db,
        "SELECT COUNT(*) FROM (
            SELECT exchange, symbol, time FROM candles
            WHERE symbol = 'SYM000USDT'
            GROUP BY exchange, symbol, time
            HAVING COUNT(*) > 1
        )",
    );
    let expected_per_sym = 28 * 24 * 12;
    eprintln!(
        "SYM000USDT dupe groups: {} (expected {})",
        per_sym, expected_per_sym
    );
    assert_eq!(per_sym, expected_per_sym, "Per-symbol GROUP BY wrong!");

    // Global GROUP BY: should have 30 * expected_per_sym dupe groups
    let global = count(
        &db,
        "SELECT COUNT(*) FROM (
            SELECT exchange, symbol, time FROM candles
            GROUP BY exchange, symbol, time
            HAVING COUNT(*) > 1
        )",
    );
    eprintln!(
        "Global dupe groups: {} (expected {})",
        global, expected_pairs
    );
    assert_eq!(
        global, expected_pairs,
        "Global GROUP BY HAVING is wrong! Expected {}, got {}",
        expected_pairs, global
    );
}

/// GROUP BY test with close+reopen (no explicit checkpoint).
/// The close seals all hot to cold. Reopen loads from cold.
#[test]
fn test_group_by_having_after_close_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}/group_by_reopen?checkpoint_interval=0",
        dir.path().display()
    );

    // Phase 1: Insert duplicate rows, close DB (triggers seal on close)
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE candles (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                time TIMESTAMP NOT NULL,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open FLOAT
            )",
            (),
        )
        .unwrap();

        let symbols: Vec<String> = (0..30).map(|i| format!("SYM{:03}USDT", i)).collect();
        let stmt = db
            .prepare(
                "INSERT INTO candles (time, exchange, symbol, open) VALUES ($1, 'binance', $2, $3)",
            )
            .unwrap();

        db.execute("BEGIN", ()).unwrap();
        for sym in &symbols {
            for day in 1..=28 {
                for hour in 0..24 {
                    for minute in (0..60).step_by(5) {
                        let ts = format!("2026-02-{:02} {:02}:{:02}:00", day, hour, minute);
                        stmt.execute((ts.as_str(), sym.as_str(), 100.0f64)).unwrap();
                        stmt.execute((ts.as_str(), sym.as_str(), 100.0f64)).unwrap();
                    }
                }
            }
        }
        db.execute("COMMIT", ()).unwrap();

        let total = count(&db, "SELECT COUNT(*) FROM candles");
        let expected = 30 * 28 * 24 * 12 * 2;
        eprintln!("[phase1] Total: {} (expected {})", total, expected);
        assert_eq!(total, expected as i64);

        // Close WITHOUT explicit checkpoint — close_engine does force seal
        db.close().unwrap();
    }

    // Phase 2: Reopen and check GROUP BY on cold data
    {
        let db = Database::open(&dsn).unwrap();

        let total = count(&db, "SELECT COUNT(*) FROM candles");
        let expected = 30 * 28 * 24 * 12 * 2;
        eprintln!(
            "[phase2] Total after reopen: {} (expected {})",
            total, expected
        );
        assert_eq!(total, expected as i64, "Reopen changed count!");

        // Per-symbol GROUP BY
        let per_sym = count(
            &db,
            "SELECT COUNT(*) FROM (
                SELECT exchange, symbol, time FROM candles
                WHERE symbol = 'SYM000USDT'
                GROUP BY exchange, symbol, time
                HAVING COUNT(*) > 1
            )",
        );
        let expected_per_sym = 28 * 24 * 12;
        eprintln!(
            "[phase2] SYM000USDT dupes: {} (expected {})",
            per_sym, expected_per_sym
        );
        assert_eq!(
            per_sym, expected_per_sym as i64,
            "Per-symbol GROUP BY wrong!"
        );

        // Global GROUP BY
        let global = count(
            &db,
            "SELECT COUNT(*) FROM (
                SELECT exchange, symbol, time FROM candles
                GROUP BY exchange, symbol, time
                HAVING COUNT(*) > 1
            )",
        );
        let expected_global = 30 * 28 * 24 * 12;
        eprintln!(
            "[phase2] Global dupes: {} (expected {})",
            global, expected_global
        );
        assert_eq!(
            global, expected_global as i64,
            "Global GROUP BY wrong on cold data! Expected {}, got {}",
            expected_global, global
        );

        db.close().unwrap();
    }
}

#[test]
fn test_delete_cold_rows_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/dataloss3", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)", ())
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO t VALUES ($1, $2)").unwrap();
        for i in 0..100_000i64 {
            stmt.execute((i, i)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Delete 1000 cold rows
        db.execute("DELETE FROM t WHERE id < 1000", ()).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM t"), 99_000);

        db.close().unwrap();
    }
    {
        let db = Database::open(&dsn).unwrap();
        let c = count(&db, "SELECT COUNT(*) FROM t");
        assert_eq!(c, 99_000, "Expected 99000 after delete+reopen, got {}", c);
        db.close().unwrap();
    }
}
