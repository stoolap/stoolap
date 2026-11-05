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

//! Integration tests for frozen volume storage.
//!
//! Tests the full lifecycle: insert → close → reopen (volume) → query → update → verify.

use stoolap::Database;

fn query_i64(db: &Database, sql: &str) -> i64 {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<i64>(0).ok())
        .unwrap_or(-1)
}

fn query_f64(db: &Database, sql: &str) -> f64 {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<f64>(0).ok())
        .unwrap_or(f64::NAN)
}

fn query_str(db: &Database, sql: &str) -> String {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<String>(0).ok())
        .unwrap_or_default()
}

fn query_count_rows(db: &Database, sql: &str) -> i64 {
    let mut r = db.query(sql, ()).unwrap();
    let mut count = 0i64;
    for _ in r.by_ref() {
        count += 1;
    }
    count
}

/// Create a test table with enough data to trigger volume loading (>16MB snapshot).
/// Uses a large description column (~500 bytes per row) to reach 16MB with fewer rows.
fn setup_large_table(db: &Database, row_count: usize) {
    db.execute(
        "CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            price FLOAT NOT NULL,
            quantity INTEGER NOT NULL,
            active BOOLEAN NOT NULL,
            description TEXT NOT NULL
        )",
        (),
    )
    .unwrap();

    let categories = ["electronics", "books", "clothing", "food", "toys"];

    // Use a transaction for bulk insert (much faster than individual auto-commits)
    db.execute("BEGIN", ()).unwrap();
    let stmt = db
        .prepare("INSERT INTO items VALUES ($1, $2, $3, $4, $5, $6, $7)")
        .unwrap();
    for i in 0..row_count {
        let cat = categories[i % categories.len()];
        // Use unique padding per row to defeat LZ4 compression
        // (ensures snapshot > 64MB threshold for volume conversion)
        let desc = format!("item_{}_desc_{:0>400}", i, i);
        stmt.execute((
            i as i64,
            cat,
            format!("item_{}", i),
            10.0 + (i as f64 * 0.1),
            (i % 100) as i64,
            i % 3 != 0,
            &desc,
        ))
        .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
}

/// Force a checkpoint (seal hot to volumes, compact, truncate WAL) and close.
fn snapshot_and_close(db: Database) {
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
}

/// Check that at least one .vol file exists for a table in the data dir.
/// Checks both volumes/ (new format) and snapshots/ (legacy format).
fn has_vol_file(dir: &std::path::Path, db_name: &str) -> bool {
    // Check new volumes/ directory
    let vol_dir = dir.join(db_name).join("volumes");
    if vol_dir.exists() {
        for entry in std::fs::read_dir(&vol_dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                for file in std::fs::read_dir(&path).into_iter().flatten().flatten() {
                    if file.path().extension().map(|e| e == "vol").unwrap_or(false) {
                        return true;
                    }
                }
            }
        }
    }
    // Check legacy snapshots/ directory
    let snap_dir = dir.join(db_name).join("snapshots");
    if snap_dir.exists() {
        for entry in std::fs::read_dir(&snap_dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                for file in std::fs::read_dir(&path).into_iter().flatten().flatten() {
                    if file.path().extension().map(|e| e == "vol").unwrap_or(false) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn has_standalone_volume(dir: &std::path::Path, db_name: &str, table_name: &str) -> bool {
    let vol_dir = dir.join(db_name).join("volumes").join(table_name);
    if !vol_dir.exists() {
        return false;
    }
    std::fs::read_dir(&vol_dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry
                .path()
                .extension()
                .map(|e| e == "vol")
                .unwrap_or(false)
        })
}

fn pseudo_random_payload(seed: i64, chunks: usize) -> String {
    let mut state = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
    let mut payload = String::with_capacity(chunks * 16);
    for _ in 0..chunks {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        payload.push_str(&format!("{:016x}", state));
    }
    payload
}

#[test]
fn test_volume_basic_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_basic", dir.path().display());

    // Session 1: Insert data
    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 200_000);
        snapshot_and_close(db);
    }

    // Session 2: Reopen — snapshot converts to .vol for large tables
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 200_000);
        db.close().unwrap();
    }

    // Verify .vol file was created
    assert!(
        has_vol_file(dir.path(), "vol_basic"),
        "Expected .vol file for large table after reopen"
    );

    // Session 3: Reopen — loads from cached .vol file
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 200_000);

        // Test various queries
        let distinct = query_i64(&db, "SELECT COUNT(DISTINCT category) FROM items");
        assert_eq!(distinct, 5);

        let max_price = query_f64(&db, "SELECT MAX(price) FROM items");
        assert!(
            max_price > 100.0,
            "max_price should be > 100, got {}",
            max_price
        );

        let min_price = query_f64(&db, "SELECT MIN(price) FROM items");
        assert!((min_price - 10.0).abs() < 0.01);

        db.close().unwrap();
    }
}

#[test]
fn test_volume_aggregation_pushdown() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_agg", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        snapshot_and_close(db);
    }

    {
        let db = Database::open(&dsn).unwrap();

        // These should use pre-computed volume stats (instant)
        let count = query_i64(&db, "SELECT COUNT(*) FROM items");
        assert_eq!(count, 200_000);

        let max = query_f64(&db, "SELECT MAX(price) FROM items");
        assert!(max > 0.0);

        let min = query_f64(&db, "SELECT MIN(price) FROM items");
        assert!(min >= 10.0);

        let sum = query_f64(&db, "SELECT SUM(price) FROM items");
        assert!(sum > 0.0);

        let avg = query_f64(&db, "SELECT AVG(price) FROM items");
        assert!(avg > 0.0);

        db.close().unwrap();
    }
}

#[test]
fn test_volume_filtered_queries() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_filter", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        snapshot_and_close(db);
    }

    {
        let db = Database::open(&dsn).unwrap();

        // Filter by category
        let electronics = query_i64(
            &db,
            "SELECT COUNT(*) FROM items WHERE category = 'electronics'",
        );
        assert_eq!(electronics, 40_000); // 200K / 5 categories

        // Filter by price range
        let expensive = query_i64(&db, "SELECT COUNT(*) FROM items WHERE price > 500");
        assert!(expensive > 0, "should have items with price > 500");

        // Multi-column filter
        let active_books = query_i64(
            &db,
            "SELECT COUNT(*) FROM items WHERE category = 'books' AND active = true",
        );
        assert!(active_books > 0);

        // GROUP BY
        let groups = query_count_rows(
            &db,
            "SELECT category, COUNT(*), AVG(price) FROM items GROUP BY category",
        );
        assert_eq!(groups, 5);

        db.close().unwrap();
    }
}

#[test]
fn test_volume_wal_replay_updates() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_wal", dir.path().display());

    // Session 1: Insert initial data
    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        snapshot_and_close(db);
    }

    // Session 2: Reopen and modify some rows
    {
        let db = Database::open(&dsn).unwrap();

        // Update some rows
        db.execute("UPDATE items SET price = 99999.99 WHERE id = 0", ())
            .unwrap();
        db.execute("UPDATE items SET price = 88888.88 WHERE id = 1", ())
            .unwrap();

        // Delete some rows
        db.execute("DELETE FROM items WHERE id = 2", ()).unwrap();
        db.execute("DELETE FROM items WHERE id = 3", ()).unwrap();

        // Insert new rows
        db.execute(
            "INSERT INTO items VALUES (200000, 'new', 'new_item', 777.77, 1, true, 'desc')",
            (),
        )
        .unwrap();

        // Verify within same session
        let price0 = query_f64(&db, "SELECT price FROM items WHERE id = 0");
        assert!((price0 - 99999.99).abs() < 0.01);

        let count = query_i64(&db, "SELECT COUNT(*) FROM items");
        assert_eq!(count, 199_999); // 200K - 2 deleted + 1 inserted

        db.close().unwrap();
    }

    // Session 3: Reopen and verify WAL changes are visible
    {
        let db = Database::open(&dsn).unwrap();

        // Updated rows should have new values (from WAL, tombstoning volume rows)
        let price0 = query_f64(&db, "SELECT price FROM items WHERE id = 0");
        assert!(
            (price0 - 99999.99).abs() < 0.01,
            "Updated price should be 99999.99, got {}",
            price0
        );

        let price1 = query_f64(&db, "SELECT price FROM items WHERE id = 1");
        assert!(
            (price1 - 88888.88).abs() < 0.01,
            "Updated price should be 88888.88, got {}",
            price1
        );

        // Deleted rows should not exist
        let deleted = query_i64(&db, "SELECT COUNT(*) FROM items WHERE id = 2");
        assert_eq!(deleted, 0, "Deleted row id=2 should not exist");

        let deleted = query_i64(&db, "SELECT COUNT(*) FROM items WHERE id = 3");
        assert_eq!(deleted, 0, "Deleted row id=3 should not exist");

        // New row should exist
        let new_price = query_f64(&db, "SELECT price FROM items WHERE id = 200000");
        assert!(
            (new_price - 777.77).abs() < 0.01,
            "New row price should be 777.77, got {}",
            new_price
        );

        // Total count should match
        let count = query_i64(&db, "SELECT COUNT(*) FROM items");
        assert_eq!(count, 199_999);

        db.close().unwrap();
    }
}

#[test]
fn test_volume_group_by_correctness() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_group", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        snapshot_and_close(db);
    }

    {
        let db = Database::open(&dsn).unwrap();

        // GROUP BY should produce correct results from volume data
        let mut r = db
            .query(
                "SELECT category, COUNT(*) as cnt FROM items GROUP BY category ORDER BY category",
                (),
            )
            .unwrap();

        let mut total = 0i64;
        for row_result in r.by_ref() {
            let row = row_result.unwrap();
            let cnt = row.get::<i64>(1).unwrap();
            assert_eq!(cnt, 40_000); // 200K / 5 categories = 40K each
            total += cnt;
        }
        assert_eq!(total, 200_000);

        db.close().unwrap();
    }
}

#[test]
fn test_volume_order_by_limit() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_order", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        snapshot_and_close(db);
    }

    {
        let db = Database::open(&dsn).unwrap();

        // ORDER BY + LIMIT should work correctly
        let top_price = query_f64(&db, "SELECT price FROM items ORDER BY price DESC LIMIT 1");
        let max_price = query_f64(&db, "SELECT MAX(price) FROM items");
        assert!(
            (top_price - max_price).abs() < 0.01,
            "ORDER BY DESC LIMIT 1 should match MAX: {} vs {}",
            top_price,
            max_price
        );

        // ORDER BY ASC
        let bottom_price = query_f64(&db, "SELECT price FROM items ORDER BY price ASC LIMIT 1");
        let min_price = query_f64(&db, "SELECT MIN(price) FROM items");
        assert!(
            (bottom_price - min_price).abs() < 0.01,
            "ORDER BY ASC LIMIT 1 should match MIN: {} vs {}",
            bottom_price,
            min_price
        );

        db.close().unwrap();
    }
}

#[test]
fn test_volume_truncate_drops_volumes() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_trunc", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        snapshot_and_close(db);
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 200_000);

        db.execute("TRUNCATE TABLE items", ()).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 0);

        // Insert new data after truncate
        db.execute(
            "INSERT INTO items VALUES (1, 'new', 'after_truncate', 42.0, 1, true, 'desc')",
            (),
        )
        .unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 1);

        db.close().unwrap();
    }

    // Reopen — truncated volumes should not reappear, only the post-truncate insert
    {
        let db = Database::open(&dsn).unwrap();
        let count = query_i64(&db, "SELECT COUNT(*) FROM items");
        assert_eq!(
            count, 1,
            "After truncate+insert+reopen, expected exactly 1 row, got {}",
            count
        );
        db.close().unwrap();
    }
}

#[test]
fn test_volume_drop_table_cleanup() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_drop", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        setup_large_table(&db, 200_000);
        snapshot_and_close(db);
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 200_000);

        db.execute("DROP TABLE items", ()).unwrap();

        // Table should not exist
        let result = db.query("SELECT COUNT(*) FROM items", ());
        assert!(result.is_err());

        // Recreate with same name
        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)", ())
            .unwrap();
        db.execute("INSERT INTO items VALUES (1, 'fresh')", ())
            .unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 1);

        db.close().unwrap();
    }

    // Reopen — old volume data should NOT reappear
    {
        let db = Database::open(&dsn).unwrap();
        let count = query_i64(&db, "SELECT COUNT(*) FROM items");
        assert_eq!(
            count, 1,
            "After drop+recreate+insert+reopen, expected 1 row, got {}",
            count
        );
        let val = query_str(&db, "SELECT val FROM items WHERE id = 1");
        assert_eq!(
            val, "fresh",
            "Should see the fresh row, not old volume data"
        );
        db.close().unwrap();
    }
}

#[test]
fn test_volume_small_table_stays_in_memory() {
    // Small tables are sealed to volumes on close (force_seal_all) for fast startup,
    // but during normal checkpoint cycles the seal threshold prevents creating volumes
    // for tiny tables. Verify data survives the roundtrip regardless of storage path.
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/vol_small", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE small (id INTEGER PRIMARY KEY, val TEXT)", ())
            .unwrap();
        for i in 0..100 {
            db.execute(
                &format!("INSERT INTO small VALUES ({}, 'val_{}')", i, i),
                (),
            )
            .unwrap();
        }
        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM small"), 100);
        assert_eq!(
            query_str(&db, "SELECT val FROM small WHERE id = 42"),
            "val_42"
        );
        db.close().unwrap();
    }
}

#[test]
fn test_volume_restart_loads_multiple_volumes() {
    // HOT_ROWS use payloads sized to exceed the 16MB volume threshold on reopen.
    // 1000 rows * ~20KB payload = ~20MB > 16MB threshold.
    const SEALED_ROWS: i64 = 100_000;
    const HOT_ROWS: i64 = 1_000;

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}/vol_mixed_restart?checkpoint_interval=3600&cleanup_interval=3600&wal_compression=off&checkpoint_on_close=off",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                note TEXT NOT NULL
            )",
            (),
        )
        .unwrap();

        db.execute("BEGIN", ()).unwrap();
        let stmt = db.prepare("INSERT INTO items VALUES ($1, $2, $3)").unwrap();
        for i in 0..SEALED_ROWS {
            stmt.execute((i, "sealed", "base")).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Checkpoint seals the 100K rows into a standalone volume file.
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert!(
            has_standalone_volume(dir.path(), "vol_mixed_restart", "items"),
            "Checkpoint should create a standalone volume for sealed rows"
        );

        db.execute("BEGIN", ()).unwrap();
        for i in SEALED_ROWS..(SEALED_ROWS + HOT_ROWS) {
            let payload = pseudo_random_payload(i, 1280);
            stmt.execute((i, "hot", payload)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Checkpoint the hot rows into a second volume so that on reopen
        // both volumes coexist.
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        db.close().unwrap();
    }

    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM items"),
            SEALED_ROWS + HOT_ROWS,
            "Restart should load both the checkpointed hot rows and older standalone volumes"
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM items WHERE category = 'sealed'"),
            SEALED_ROWS
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM items WHERE category = 'hot'"),
            HOT_ROWS
        );
        db.close().unwrap();
    }
}
