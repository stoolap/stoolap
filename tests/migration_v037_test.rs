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

//! Migration test: v0.3.7 (snapshot-based) -> volume-based storage.
//!
//! Uses a real v0.3.7 database created by `tests/testdata/create_v037db.sql`
//! with the snapshot-based engine. The test copies it to a temp directory,
//! opens it with the current (volume-based) engine, and verifies:
//!   1. All data survives migration (snapshot + WAL replay)
//!   2. snapshots/ directory is removed after migration
//!   3. volumes/ directory is created
//!   4. All queries produce correct results
//!   5. New writes work after migration
//!   6. Data persists across another close/reopen cycle

use stoolap::Database;

/// Copy the v0.3.7 test database to a fresh temp directory.
fn copy_v037db(dir: &std::path::Path) {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("testdata")
        .join("v037db");
    assert!(src.exists(), "v037db fixture missing at {:?}", src);
    copy_dir_recursive(&src, dir);
}

fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let ty = entry.file_type().unwrap();
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path);
        } else {
            std::fs::copy(entry.path(), &dest_path).unwrap();
        }
    }
}

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

fn query_str(db: &Database, sql: &str) -> Option<String> {
    let mut r = db.query(sql, ()).unwrap();
    r.next()
        .and_then(|r| r.ok())
        .and_then(|r| r.get::<String>(0).ok())
}

fn query_rows(db: &Database, sql: &str) -> Vec<Vec<String>> {
    let mut r = db.query(sql, ()).unwrap();
    let mut rows = Vec::new();
    while let Some(Ok(row)) = r.next() {
        let mut vals = Vec::new();
        for i in 0..row.columns().len() {
            vals.push(row.get::<String>(i).unwrap_or_else(|_| "NULL".to_string()));
        }
        rows.push(vals);
    }
    rows
}

// ── Test: full migration from v0.3.7 snapshot-based DB ──────────────

#[test]
fn test_migration_v037_data_integrity() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated");
    copy_v037db(&db_dir);

    // Confirm legacy layout: snapshots/ exists, volumes/ does not
    assert!(db_dir.join("snapshots").exists(), "should have snapshots/");
    assert!(
        !db_dir.join("volumes").exists(),
        "should NOT have volumes/ yet"
    );

    let dsn = format!("file://{}", db_dir.display());

    // ── Session 1: Open triggers migration ──────────────────────────
    {
        let db = Database::open(&dsn).unwrap();

        // Row counts (snapshot data + WAL replay)
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 13);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 22);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM products"), 11);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM events"), 53);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM documents"), 8);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM vectors"), 16);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 26);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nullable_types"), 11);

        // Pre-snapshot changes: Alice balance updated, Eve and Mia deleted
        let alice_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 1");
        assert!(
            (alice_balance - 1100.0).abs() < 0.01,
            "Alice balance should be 1100.0, got {}",
            alice_balance
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM users WHERE id = 5"),
            0,
            "Eve (id=5) should be deleted"
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM users WHERE id = 13"),
            0,
            "Mia (id=13) should be deleted"
        );

        // Post-snapshot WAL changes
        // Bob balance updated to 5000 after snapshot
        let bob_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 2");
        assert!(
            (bob_balance - 5000.0).abs() < 0.01,
            "Bob balance should be 5000.0 (WAL update), got {}",
            bob_balance
        );

        // Charlie (id=3) deleted after snapshot
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM users WHERE id = 3"),
            0,
            "Charlie (id=3) should be deleted (WAL)"
        );

        // Quinn (id=16) inserted after snapshot
        let quinn = query_str(&db, "SELECT name FROM users WHERE id = 16");
        assert_eq!(quinn.as_deref(), Some("Quinn"));

        // Order 4 status updated to 'shipped' after snapshot
        let status = query_str(&db, "SELECT status FROM orders WHERE id = 4");
        assert_eq!(status.as_deref(), Some("shipped"));

        // New orders 21, 22 from WAL
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM orders WHERE id >= 21"),
            2
        );

        // New product from WAL
        let nut = query_str(&db, "SELECT name FROM products WHERE sku = 'NUT-001'");
        assert_eq!(nut.as_deref(), Some("Nut Pack"));

        // New vector from WAL
        let snake = query_str(&db, "SELECT label FROM vectors WHERE id = 16");
        assert_eq!(snake.as_deref(), Some("snake"));

        // Post-snapshot events
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM events WHERE id >= 51"),
            3
        );

        // Aggregation checks
        let sum_balance = query_f64(&db, "SELECT SUM(balance) FROM users WHERE active = true");
        assert!(
            (sum_balance - 28551.05).abs() < 0.1,
            "SUM(balance) should be ~28551.05, got {}",
            sum_balance
        );

        let max_severity = query_i64(&db, "SELECT MAX(severity) FROM events");
        assert_eq!(max_severity, 5);

        let distinct_cats = query_i64(&db, "SELECT COUNT(DISTINCT category) FROM products");
        assert_eq!(distinct_cats, 4);

        // JSON access
        let role = query_str(
            &db,
            "SELECT JSON_EXTRACT(metadata, '$.role') FROM users WHERE id = 1",
        );
        assert_eq!(role.as_deref(), Some("admin"));

        // Joins
        let join_count = query_i64(
            &db,
            "SELECT COUNT(*) FROM users u JOIN orders o ON u.id = o.user_id",
        );
        assert!(join_count > 0, "JOIN should return rows");

        // GROUP BY
        let groups = query_rows(
            &db,
            "SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY category",
        );
        assert_eq!(groups.len(), 4);

        // Nullable edge cases
        let null_count = query_i64(
            &db,
            "SELECT COUNT(*) FROM nullable_types WHERE int_val IS NULL",
        );
        assert!(null_count >= 2);

        db.close().unwrap();
    }

    // Verify migration happened: snapshots/ removed, volumes/ created
    assert!(
        !db_dir.join("snapshots").exists(),
        "snapshots/ should be removed after migration"
    );
    let vol_dir = db_dir.join("volumes");
    assert!(vol_dir.exists(), "volumes/ should exist after migration");

    // Verify each table has a volume directory with at least one .vol file
    let expected_tables = [
        "users",
        "orders",
        "products",
        "events",
        "documents",
        "vectors",
        "metrics",
        "nullable_types",
    ];
    for table in &expected_tables {
        let table_vol_dir = vol_dir.join(table);
        assert!(
            table_vol_dir.exists(),
            "volumes/{} directory should exist",
            table
        );
        let has_vol = std::fs::read_dir(&table_vol_dir)
            .unwrap()
            .flatten()
            .any(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "vol")
                    .unwrap_or(false)
            });
        assert!(
            has_vol,
            "volumes/{} should contain at least one .vol file",
            table
        );
        // Each table should also have a manifest
        let has_manifest = std::fs::read_dir(&table_vol_dir)
            .unwrap()
            .flatten()
            .any(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n == "manifest.bin")
                    .unwrap_or(false)
            });
        assert!(has_manifest, "volumes/{} should have a manifest.bin", table);
    }

    // ── Session 2: Reopen migrated DB, verify + write new data ──────
    {
        let db = Database::open(&dsn).unwrap();

        // All 8 tables still intact from volumes
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 13);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 22);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM products"), 11);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM events"), 53);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM documents"), 8);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM vectors"), 16);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 26);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nullable_types"), 11);

        // Spot-check values survived the volume roundtrip
        let alice_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 1");
        assert!(
            (alice_balance - 1100.0).abs() < 0.01,
            "Alice balance should survive volume roundtrip, got {}",
            alice_balance
        );
        let bob_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 2");
        assert!(
            (bob_balance - 5000.0).abs() < 0.01,
            "Bob WAL update should survive volume roundtrip, got {}",
            bob_balance
        );
        let quinn = query_str(&db, "SELECT name FROM users WHERE id = 16");
        assert_eq!(quinn.as_deref(), Some("Quinn"), "WAL insert should survive");

        // Index lookups on volume data
        let email_hit = query_str(
            &db,
            "SELECT name FROM users WHERE email = 'ivy@example.com'",
        );
        assert_eq!(
            email_hit.as_deref(),
            Some("Ivy"),
            "Hash index lookup should work on volumes"
        );

        let age_range = query_i64(
            &db,
            "SELECT COUNT(*) FROM users WHERE age BETWEEN 30 AND 40",
        );
        assert!(
            age_range > 0,
            "BTree index range scan should work on volumes"
        );

        let sku_hit = query_str(&db, "SELECT name FROM products WHERE sku = 'WDG-001'");
        assert_eq!(
            sku_hit.as_deref(),
            Some("Widget"),
            "Unique index lookup should work"
        );

        // Write new data on the migrated DB
        db.execute(
            "INSERT INTO users VALUES (17, 'Rose', 'rose@example.com', 29, 3000.00, true, '2025-03-01T10:00:00Z', '{\"role\": \"user\"}')",
            (),
        )
        .unwrap();

        db.execute("UPDATE users SET balance = 9999.99 WHERE id = 10", ())
            .unwrap();

        db.execute("DELETE FROM orders WHERE id = 22", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 14);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 21);

        let rose = query_str(&db, "SELECT name FROM users WHERE id = 17");
        assert_eq!(rose.as_deref(), Some("Rose"));

        let jack_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 10");
        assert!(
            (jack_balance - 9999.99).abs() < 0.01,
            "Jack balance should be 9999.99, got {}",
            jack_balance
        );

        db.close().unwrap();
    }

    // Verify no regression: snapshots/ must NOT reappear, volumes/ still there
    assert!(
        !db_dir.join("snapshots").exists(),
        "snapshots/ must not reappear after session 2"
    );
    assert!(
        db_dir.join("volumes").exists(),
        "volumes/ should still exist after session 2"
    );

    // ── Session 3: Final reopen to verify persistence ───────────────
    {
        let db = Database::open(&dsn).unwrap();

        // All tables: original migrated data + session 2 changes
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 14);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 21);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM products"), 11);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM events"), 53);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM documents"), 8);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM vectors"), 16);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 26);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nullable_types"), 11);

        // Session 2 writes persisted
        let rose = query_str(&db, "SELECT name FROM users WHERE id = 17");
        assert_eq!(rose.as_deref(), Some("Rose"));

        let jack_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 10");
        assert!(
            (jack_balance - 9999.99).abs() < 0.01,
            "Jack balance should be 9999.99 after reopen, got {}",
            jack_balance
        );

        // Session 2 delete persisted
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM orders WHERE id = 22"),
            0,
            "Order 22 should stay deleted"
        );

        // Original migrated data still intact
        let quinn = query_str(&db, "SELECT name FROM users WHERE id = 16");
        assert_eq!(quinn.as_deref(), Some("Quinn"));

        let bob_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 2");
        assert!(
            (bob_balance - 5000.0).abs() < 0.01,
            "Bob balance should still be 5000.0, got {}",
            bob_balance
        );

        let alice_balance = query_f64(&db, "SELECT balance FROM users WHERE id = 1");
        assert!(
            (alice_balance - 1100.0).abs() < 0.01,
            "Alice balance should still be 1100.0, got {}",
            alice_balance
        );

        // Deleted users stay deleted
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users WHERE id = 3"), 0);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users WHERE id = 5"), 0);
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM users WHERE id = 13"),
            0
        );

        // Vector data intact
        assert_eq!(
            query_i64(
                &db,
                "SELECT COUNT(*) FROM vectors WHERE category = 'animal'"
            ),
            7
        );

        // Index lookups still work
        let email_hit = query_str(
            &db,
            "SELECT name FROM users WHERE email = 'rose@example.com'",
        );
        assert_eq!(email_hit.as_deref(), Some("Rose"));

        let sku_hit = query_str(&db, "SELECT name FROM products WHERE sku = 'NUT-001'");
        assert_eq!(sku_hit.as_deref(), Some("Nut Pack"));

        // Aggregations on final state
        let max_sev = query_i64(&db, "SELECT MAX(severity) FROM events");
        assert_eq!(max_sev, 5);

        let total_products = query_f64(&db, "SELECT SUM(price) FROM products");
        assert!(total_products > 0.0);

        db.close().unwrap();
    }

    // Final check: still no snapshots/, volumes/ intact
    assert!(
        !db_dir.join("snapshots").exists(),
        "snapshots/ must not reappear after session 3"
    );
    assert!(
        db_dir.join("volumes").exists(),
        "volumes/ should still exist after session 3"
    );
}

// ── Test: index survival after migration ────────────────────────────

#[test]
fn test_migration_v037_indexes() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_idx");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());

    let db = Database::open(&dsn).unwrap();

    // Index-assisted lookups should work
    let email_lookup = query_str(
        &db,
        "SELECT name FROM users WHERE email = 'bob@example.com'",
    );
    assert_eq!(email_lookup.as_deref(), Some("Bob"));

    let age_range = query_i64(
        &db,
        "SELECT COUNT(*) FROM users WHERE age BETWEEN 25 AND 35",
    );
    assert!(age_range > 0);

    // Timestamp index lookup
    let ts_count = query_i64(
        &db,
        "SELECT COUNT(*) FROM orders WHERE created_at > '2025-02-01T00:00:00Z'",
    );
    assert!(ts_count > 0);

    // Unique index (sku)
    let sku_lookup = query_str(&db, "SELECT name FROM products WHERE sku = 'GDG-001'");
    assert_eq!(sku_lookup.as_deref(), Some("Gadget"));

    // Unique constraint still enforced
    let dup = db.execute(
        "INSERT INTO products VALUES (99, 'GDG-001', 'Duplicate', null, 1.0, null, true, 'x', null, null)",
        (),
    );
    assert!(
        dup.is_err(),
        "Unique constraint on sku should prevent duplicate"
    );

    db.close().unwrap();
}

// ── Test: aggregation correctness after migration ───────────────────

#[test]
fn test_migration_v037_aggregations() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_agg");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // SUM
    let total_balance = query_f64(&db, "SELECT SUM(balance) FROM users");
    assert!(total_balance > 20000.0, "SUM(balance) = {}", total_balance);

    // AVG
    let avg_amount = query_f64(&db, "SELECT AVG(amount) FROM orders");
    assert!(
        (avg_amount - 49.377727).abs() < 0.01,
        "AVG(amount) = {}",
        avg_amount
    );

    // MIN / MAX
    let min_price = query_f64(&db, "SELECT MIN(price) FROM products");
    assert!(
        (min_price - 6.99).abs() < 0.01,
        "MIN(price) = {}",
        min_price
    );

    let max_price = query_f64(&db, "SELECT MAX(price) FROM products");
    assert!(
        (max_price - 99.99).abs() < 0.01,
        "MAX(price) = {}",
        max_price
    );

    // COUNT DISTINCT
    let distinct_hosts = query_i64(&db, "SELECT COUNT(DISTINCT host) FROM metrics");
    assert_eq!(distinct_hosts, 4); // srv-01, srv-02, srv-03, srv-04

    // GROUP BY with HAVING
    let heavy_users = query_i64(
        &db,
        "SELECT COUNT(*) FROM (SELECT user_id, COUNT(*) as cnt FROM orders GROUP BY user_id HAVING cnt >= 3)",
    );
    assert!(heavy_users > 0);

    db.close().unwrap();
}

// ── Test: vector data survives migration ────────────────────────────

#[test]
fn test_migration_v037_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_vec");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // All vectors present
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM vectors"), 16);

    // Category counts
    assert_eq!(
        query_i64(
            &db,
            "SELECT COUNT(*) FROM vectors WHERE category = 'animal'"
        ),
        7
    );
    assert_eq!(
        query_i64(
            &db,
            "SELECT COUNT(*) FROM vectors WHERE category = 'vehicle'"
        ),
        5
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM vectors WHERE category = 'plant'"),
        4
    );

    // Post-snapshot vector
    let snake_score = query_f64(&db, "SELECT score FROM vectors WHERE label = 'snake'");
    assert!((snake_score - 0.86).abs() < 0.01);

    db.close().unwrap();
}

// ── Test: NULL handling after migration ──────────────────────────────

#[test]
fn test_migration_v037_nulls() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_null");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nullable_types"), 11);

    // All-null rows
    let all_null = query_i64(
        &db,
        "SELECT COUNT(*) FROM nullable_types WHERE int_val IS NULL AND float_val IS NULL AND text_val IS NULL AND bool_val IS NULL AND ts_val IS NULL AND json_val IS NULL",
    );
    assert_eq!(all_null, 2, "Should have 2 all-null rows (id=2, id=8)");

    // Edge values
    let max_int = query_i64(&db, "SELECT int_val FROM nullable_types WHERE id = 5");
    assert_eq!(max_int, 2147483647);

    let min_int = query_i64(&db, "SELECT int_val FROM nullable_types WHERE id = 9");
    assert_eq!(min_int, -2147483648);

    // Empty string vs NULL
    let empty = query_str(&db, "SELECT text_val FROM nullable_types WHERE id = 3");
    assert_eq!(empty.as_deref(), Some(""));

    // Post-snapshot nullable row
    let post = query_str(&db, "SELECT text_val FROM nullable_types WHERE id = 11");
    assert_eq!(post.as_deref(), Some("post-snapshot"));

    db.close().unwrap();
}

// ── Test: timestamp values survive migration with precision ──────────

#[test]
fn test_migration_v037_timestamps() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_ts");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // Verify exact timestamp values from users table
    let ts = query_str(&db, "SELECT created_at FROM users WHERE id = 1");
    assert!(
        ts.as_deref().unwrap().starts_with("2024-01-15"),
        "User 1 created_at should be 2024-01-15, got {:?}",
        ts
    );

    // Verify timestamp from orders
    let order_ts = query_str(&db, "SELECT created_at FROM orders WHERE id = 1");
    assert!(
        order_ts.as_deref().unwrap().starts_with("2025-01-15"),
        "Order 1 created_at should be 2025-01-15, got {:?}",
        order_ts
    );

    // Verify timestamps from events
    let event_ts = query_str(&db, "SELECT occurred_at FROM events WHERE id = 1");
    assert!(
        event_ts.as_deref().unwrap().starts_with("2025-01-01"),
        "Event 1 occurred_at should be 2025-01-01, got {:?}",
        event_ts
    );

    // Post-snapshot timestamp (WAL)
    let wal_ts = query_str(&db, "SELECT created_at FROM users WHERE id = 16");
    assert!(
        wal_ts.as_deref().unwrap().starts_with("2025-02-20"),
        "Quinn created_at should be 2025-02-20, got {:?}",
        wal_ts
    );

    // Post-snapshot order timestamp
    let wal_order_ts = query_str(&db, "SELECT created_at FROM orders WHERE id = 21");
    assert!(
        wal_order_ts.as_deref().unwrap().starts_with("2025-02-20"),
        "Order 21 created_at should be 2025-02-20, got {:?}",
        wal_order_ts
    );

    // Timestamp range queries
    let jan_orders = query_i64(
        &db,
        "SELECT COUNT(*) FROM orders WHERE created_at >= '2025-01-15T00:00:00Z' AND created_at < '2025-02-01T00:00:00Z'",
    );
    assert_eq!(jan_orders, 10, "Should have 10 January orders");

    let feb_orders = query_i64(
        &db,
        "SELECT COUNT(*) FROM orders WHERE created_at >= '2025-02-01T00:00:00Z' AND created_at < '2025-03-01T00:00:00Z'",
    );
    assert_eq!(feb_orders, 12, "Should have 12 February orders");

    // MIN/MAX timestamps
    let min_ts = query_str(&db, "SELECT MIN(created_at) FROM users");
    assert!(
        min_ts.as_deref().unwrap().starts_with("2024-01-15"),
        "MIN user created_at = {:?}",
        min_ts
    );

    let max_ts = query_str(&db, "SELECT MAX(created_at) FROM users");
    assert!(
        max_ts.as_deref().unwrap().starts_with("2025-02-20"),
        "MAX user created_at = {:?}",
        max_ts
    );

    // Edge timestamp from nullable_types
    let epoch = query_str(&db, "SELECT ts_val FROM nullable_types WHERE id = 3");
    assert!(
        epoch.as_deref().unwrap().starts_with("1970-01-01"),
        "Epoch timestamp = {:?}",
        epoch
    );

    let old_ts = query_str(&db, "SELECT ts_val FROM nullable_types WHERE id = 9");
    assert!(
        old_ts.as_deref().unwrap().starts_with("1900-01-01"),
        "1900 timestamp = {:?}",
        old_ts
    );

    let future_ts = query_str(&db, "SELECT ts_val FROM nullable_types WHERE id = 5");
    assert!(
        future_ts.as_deref().unwrap().starts_with("2099-12-31"),
        "2099 timestamp = {:?}",
        future_ts
    );

    // ORDER BY timestamp
    let first_event = query_str(
        &db,
        "SELECT occurred_at FROM events ORDER BY occurred_at ASC LIMIT 1",
    );
    assert!(
        first_event
            .as_deref()
            .unwrap()
            .starts_with("2025-01-01T00:00:01"),
        "First event = {:?}",
        first_event
    );

    db.close().unwrap();
}

// ── Test: HNSW vector search after migration ────────────────────────

#[test]
fn test_migration_v037_hnsw_search() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_hnsw");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // Nearest neighbor search: query near 'cat' embedding [0.1, 0.2, 0.3, 0.4]
    // Should return cat itself as closest, then similar animals
    let nearest = query_str(
        &db,
        "SELECT label FROM vectors ORDER BY embedding <=> '[0.1, 0.2, 0.3, 0.4]' LIMIT 1",
    );
    assert_eq!(
        nearest.as_deref(),
        Some("cat"),
        "Nearest to cat embedding should be cat"
    );

    // Query near vehicle space [0.8, 0.1, 0.05, 0.05]
    let nearest_vehicle = query_str(
        &db,
        "SELECT label FROM vectors ORDER BY embedding <=> '[0.8, 0.1, 0.05, 0.05]' LIMIT 1",
    );
    assert_eq!(
        nearest_vehicle.as_deref(),
        Some("car"),
        "Nearest to car embedding should be car"
    );

    // Top 3 nearest to plant space [0.3, 0.7, 0.1, 0.2]
    let top3 = query_rows(
        &db,
        "SELECT label, category FROM vectors ORDER BY embedding <=> '[0.3, 0.7, 0.1, 0.2]' LIMIT 3",
    );
    assert_eq!(top3.len(), 3);
    // All top 3 should be plants (rose is the exact match)
    assert_eq!(top3[0][0], "rose");

    // Filtered vector search
    let nearest_animal = query_str(
        &db,
        "SELECT label FROM vectors WHERE category = 'animal' ORDER BY embedding <=> '[0.1, 0.2, 0.3, 0.4]' LIMIT 1",
    );
    assert_eq!(nearest_animal.as_deref(), Some("cat"));

    // Post-snapshot vector (snake) should be searchable
    let near_snake = query_str(
        &db,
        "SELECT label FROM vectors ORDER BY embedding <=> '[0.13, 0.21, 0.33, 0.41]' LIMIT 1",
    );
    assert_eq!(near_snake.as_deref(), Some("snake"));

    db.close().unwrap();
}

// ── Test: schema correctness after migration ────────────────────────

#[test]
fn test_migration_v037_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_schema");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // NOT NULL constraints enforced
    let null_name = db.execute(
        "INSERT INTO users VALUES (99, null, 'x@x.com', 30, 100.0, true, '2025-01-01T00:00:00Z', null)",
        (),
    );
    assert!(
        null_name.is_err(),
        "NOT NULL on users.name should be enforced"
    );

    let null_email = db.execute(
        "INSERT INTO users VALUES (99, 'Test', null, 30, 100.0, true, '2025-01-01T00:00:00Z', null)",
        (),
    );
    assert!(
        null_email.is_err(),
        "NOT NULL on users.email should be enforced"
    );

    let null_active = db.execute(
        "INSERT INTO users VALUES (99, 'Test', 'x@x.com', 30, 100.0, null, '2025-01-01T00:00:00Z', null)",
        (),
    );
    assert!(
        null_active.is_err(),
        "NOT NULL on users.active should be enforced"
    );

    // PRIMARY KEY uniqueness enforced
    let dup_pk = db.execute(
        "INSERT INTO users VALUES (1, 'Dup', 'dup@x.com', 20, 0.0, true, '2025-01-01T00:00:00Z', null)",
        (),
    );
    assert!(
        dup_pk.is_err(),
        "PRIMARY KEY uniqueness on users.id should be enforced"
    );

    let dup_order_pk = db.execute(
        "INSERT INTO orders VALUES (1, 1, 'X', 1.0, 1, 'pending', '2025-01-01T00:00:00Z', null)",
        (),
    );
    assert!(
        dup_order_pk.is_err(),
        "PRIMARY KEY on orders.id should be enforced"
    );

    // UNIQUE index enforced
    let dup_sku = db.execute(
        "INSERT INTO products VALUES (99, 'WDG-001', 'Dup', null, 1.0, null, true, 'x', null, null)",
        (),
    );
    assert!(
        dup_sku.is_err(),
        "UNIQUE index on products.sku should be enforced"
    );

    // Nullable columns accept NULL
    db.execute(
        "INSERT INTO products VALUES (99, 'TEST-001', 'Test', null, 1.0, null, true, 'test', null, null)",
        (),
    )
    .expect("Nullable columns should accept NULL");
    // Clean up
    db.execute("DELETE FROM products WHERE id = 99", ())
        .unwrap();

    // Verify column count per table via SELECT *
    let user_row = query_rows(&db, "SELECT * FROM users WHERE id = 1");
    assert_eq!(user_row.len(), 1);
    assert_eq!(user_row[0].len(), 8, "users should have 8 columns");

    let order_row = query_rows(&db, "SELECT * FROM orders WHERE id = 1");
    assert_eq!(order_row[0].len(), 8, "orders should have 8 columns");

    let product_row = query_rows(&db, "SELECT * FROM products WHERE id = 1");
    assert_eq!(product_row[0].len(), 10, "products should have 10 columns");

    let event_row = query_rows(&db, "SELECT * FROM events WHERE id = 1");
    assert_eq!(event_row[0].len(), 7, "events should have 7 columns");

    let doc_row = query_rows(&db, "SELECT * FROM documents WHERE id = 1");
    assert_eq!(doc_row[0].len(), 10, "documents should have 10 columns");

    let vec_row = query_rows(&db, "SELECT * FROM vectors WHERE id = 1");
    assert_eq!(vec_row[0].len(), 5, "vectors should have 5 columns");

    let metric_row = query_rows(&db, "SELECT * FROM metrics WHERE id = 1");
    assert_eq!(metric_row[0].len(), 6, "metrics should have 6 columns");

    let null_row = query_rows(&db, "SELECT * FROM nullable_types WHERE id = 1");
    assert_eq!(null_row[0].len(), 7, "nullable_types should have 7 columns");

    db.close().unwrap();
}

// ── Test: every row in users table verified ─────────────────────────

#[test]
fn test_migration_v037_users_every_row() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_allrows");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // Expected final state of users table after all changes:
    // Original: 15 users (id 1-15)
    // Pre-snapshot: UPDATE id=1 balance->1100, UPDATE id=6 active->false, DELETE id=5, DELETE id=13
    // Post-snapshot WAL: INSERT id=16, UPDATE id=2 balance->5000, DELETE id=3
    // Surviving: 1,2,4,6,7,8,9,10,11,12,14,15,16
    let expected: Vec<(i64, &str, f64, bool)> = vec![
        (1, "Alice", 1100.0, true),
        (2, "Bob", 5000.0, true),
        (4, "Diana", 3200.25, true),
        (6, "Frank", 4100.0, false), // active set to false pre-snapshot
        (7, "Grace", 780.50, true),
        (8, "Hank", 6000.0, false),
        (9, "Ivy", 920.30, true),
        (10, "Jack", 1500.0, true),
        (11, "Karen", 2200.0, true),
        (12, "Leo", 3800.0, true),
        (14, "Noah", 5500.0, true),
        (15, "Olivia", 1800.0, true),
        (16, "Quinn", 2750.0, true),
    ];

    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM users"),
        expected.len() as i64
    );

    for (id, name, balance, active) in &expected {
        let row = query_rows(
            &db,
            &format!("SELECT name, balance, active FROM users WHERE id = {}", id),
        );
        assert_eq!(row.len(), 1, "User id={} should exist", id);
        assert_eq!(row[0][0], *name, "User id={} name mismatch", id);
        let bal: f64 = row[0][1].parse().unwrap();
        assert!(
            (bal - balance).abs() < 0.01,
            "User id={} balance: expected {}, got {}",
            id,
            balance,
            bal
        );
        let act = row[0][2] == "true" || row[0][2] == "1";
        assert_eq!(
            act, *active,
            "User id={} active: expected {}, got {}",
            id, active, row[0][2]
        );
    }

    // Verify deleted users are truly gone
    for id in &[3, 5, 13] {
        assert_eq!(
            query_i64(
                &db,
                &format!("SELECT COUNT(*) FROM users WHERE id = {}", id)
            ),
            0,
            "User id={} should be deleted",
            id
        );
    }

    db.close().unwrap();
}

// ── Test: every row in orders table verified ────────────────────────

#[test]
fn test_migration_v037_orders_every_row() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_orders");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // 20 original + 2 WAL inserts = 22 orders
    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 22);

    // Verify every original order exists with correct product and amount
    let expected: Vec<(i64, i64, &str, f64, &str)> = vec![
        (1, 1, "Widget", 29.99, "completed"),
        (2, 1, "Gadget", 49.99, "completed"),
        (3, 2, "Widget", 29.99, "shipped"),
        (4, 3, "Doohickey", 99.99, "shipped"), // updated from pending via WAL
        (5, 4, "Widget", 29.99, "completed"),
        (6, 2, "Gadget", 49.99, "completed"),
        (7, 2, "Thingamajig", 15.50, "shipped"),
        (8, 6, "Doohickey", 99.99, "cancelled"),
        (9, 7, "Widget", 29.99, "completed"),
        (10, 8, "Gadget", 49.99, "pending"),
        (11, 1, "Thingamajig", 15.50, "completed"),
        (12, 4, "Doohickey", 99.99, "shipped"),
        (13, 9, "Widget", 29.99, "completed"),
        (14, 10, "Gadget", 49.99, "completed"),
        (15, 3, "Widget", 29.99, "pending"),
        (16, 11, "Gadget", 49.99, "completed"),
        (17, 12, "Doohickey", 99.99, "shipped"),
        (18, 14, "Widget", 29.99, "completed"),
        (19, 15, "Thingamajig", 15.50, "completed"),
        (20, 7, "Doohickey", 99.99, "pending"),
        (21, 16, "Widget", 29.99, "pending"),  // WAL insert
        (22, 2, "Gadget", 49.99, "completed"), // WAL insert
    ];

    for (id, user_id, product, amount, status) in &expected {
        let row = query_rows(
            &db,
            &format!(
                "SELECT user_id, product, amount, status FROM orders WHERE id = {}",
                id
            ),
        );
        assert_eq!(row.len(), 1, "Order id={} should exist", id);
        let uid: i64 = row[0][0].parse().unwrap();
        assert_eq!(uid, *user_id, "Order id={} user_id mismatch", id);
        assert_eq!(row[0][1], *product, "Order id={} product mismatch", id);
        let amt: f64 = row[0][2].parse().unwrap();
        assert!(
            (amt - amount).abs() < 0.01,
            "Order id={} amount: expected {}, got {}",
            id,
            amount,
            amt
        );
        assert_eq!(row[0][3], *status, "Order id={} status mismatch", id);
    }

    db.close().unwrap();
}

// ── Test: every product verified ────────────────────────────────────

#[test]
fn test_migration_v037_products_every_row() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_products");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM products"), 11);

    // Verify each product: (id, sku, name, price, in_stock, category)
    let expected: Vec<(i64, &str, &str, f64, bool, &str)> = vec![
        (1, "WDG-001", "Widget", 29.99, true, "hardware"),
        (2, "GDG-001", "Gadget", 49.99, true, "electronics"),
        (3, "DHK-001", "Doohickey", 99.99, true, "industrial"),
        (4, "THG-001", "Thingamajig", 15.50, true, "accessories"),
        (5, "WDG-002", "Widget Pro", 59.99, false, "hardware"),
        (6, "GDG-002", "Gadget Mini", 24.99, true, "electronics"),
        (7, "SPR-001", "Sprocket", 12.75, true, "hardware"),
        (8, "BLT-001", "Bolt Pack", 8.99, true, "hardware"),
        (9, "CBL-001", "Cable Set", 34.99, true, "electronics"),
        (10, "DHK-002", "Doohickey Lite", 49.99, true, "industrial"),
        (11, "NUT-001", "Nut Pack", 6.99, true, "hardware"), // WAL insert
    ];

    for (id, sku, name, price, in_stock, category) in &expected {
        let row = query_rows(
            &db,
            &format!(
                "SELECT sku, name, price, in_stock, category FROM products WHERE id = {}",
                id
            ),
        );
        assert_eq!(row.len(), 1, "Product id={} should exist", id);
        assert_eq!(row[0][0], *sku, "Product id={} sku mismatch", id);
        assert_eq!(row[0][1], *name, "Product id={} name mismatch", id);
        let p: f64 = row[0][2].parse().unwrap();
        assert!(
            (p - price).abs() < 0.01,
            "Product id={} price: expected {}, got {}",
            id,
            price,
            p
        );
        let stock = row[0][3] == "true" || row[0][3] == "1";
        assert_eq!(stock, *in_stock, "Product id={} in_stock mismatch", id);
        assert_eq!(row[0][4], *category, "Product id={} category mismatch", id);
    }

    db.close().unwrap();
}

// ── Test: JSON data integrity after migration ───────────────────────

#[test]
fn test_migration_v037_json_integrity() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_json");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    // User metadata JSON - verify structure survived
    let role = query_str(
        &db,
        "SELECT JSON_EXTRACT(metadata, '$.role') FROM users WHERE id = 1",
    );
    assert_eq!(role.as_deref(), Some("admin"));

    let role7 = query_str(
        &db,
        "SELECT JSON_EXTRACT(metadata, '$.role') FROM users WHERE id = 7",
    );
    assert_eq!(role7.as_deref(), Some("admin"));

    let role4 = query_str(
        &db,
        "SELECT JSON_EXTRACT(metadata, '$.role') FROM users WHERE id = 4",
    );
    assert_eq!(role4.as_deref(), Some("moderator"));

    // Updated JSON (Alice's metadata was updated pre-snapshot)
    let tags1 = query_str(
        &db,
        "SELECT JSON_EXTRACT(metadata, '$.tags') FROM users WHERE id = 1",
    );
    assert!(
        tags1.as_deref().unwrap().contains("updated"),
        "Alice tags should contain 'updated', got {:?}",
        tags1
    );

    // WAL-inserted user JSON
    let quinn_tags = query_str(
        &db,
        "SELECT JSON_EXTRACT(metadata, '$.tags') FROM users WHERE id = 16",
    );
    assert!(
        quinn_tags.as_deref().unwrap().contains("post-snapshot"),
        "Quinn tags should contain 'post-snapshot', got {:?}",
        quinn_tags
    );

    // Order notes JSON
    let priority = query_str(
        &db,
        "SELECT JSON_EXTRACT(notes, '$.priority') FROM orders WHERE id = 2",
    );
    assert_eq!(priority.as_deref(), Some("high"));

    let tracking = query_str(
        &db,
        "SELECT JSON_EXTRACT(notes, '$.tracking') FROM orders WHERE id = 3",
    );
    assert_eq!(tracking.as_deref(), Some("TR001"));

    // Nested JSON in events
    let user_id = query_str(
        &db,
        "SELECT JSON_EXTRACT(payload, '$.user_id') FROM events WHERE id = 1",
    );
    assert_eq!(user_id.as_deref(), Some("1"));

    let endpoint = query_str(
        &db,
        "SELECT JSON_EXTRACT(payload, '$.endpoint') FROM events WHERE id = 3",
    );
    assert_eq!(endpoint.as_deref(), Some("/api/users"));

    // Nested JSON in nullable_types
    let nested = query_str(
        &db,
        "SELECT JSON_EXTRACT(json_val, '$.nested.deep') FROM nullable_types WHERE id = 5",
    );
    assert_eq!(nested.as_deref(), Some("true"));

    // NULL JSON values
    let null_json = query_i64(&db, "SELECT COUNT(*) FROM orders WHERE notes IS NULL");
    assert!(null_json > 0, "Some orders should have NULL notes");

    // Product tags JSON arrays
    let tags = query_str(&db, "SELECT tags FROM products WHERE id = 1");
    assert!(
        tags.as_deref().unwrap().contains("popular"),
        "Product 1 tags should contain 'popular', got {:?}",
        tags
    );

    // NULL tags
    let null_tags = query_i64(&db, "SELECT COUNT(*) FROM products WHERE tags IS NULL");
    assert!(null_tags > 0, "Some products should have NULL tags");

    db.close().unwrap();
}

// ── Test: documents table every row ─────────────────────────────────

#[test]
fn test_migration_v037_documents_every_row() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_docs");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM documents"), 8);

    // Verify each document
    let expected: Vec<(i64, &str, &str, &str, i64, bool)> = vec![
        (1, "Getting Started Guide", "Alice", "guide", 3, true),
        (2, "API Reference v2", "Bob", "reference", 2, true),
        (3, "Architecture Overview", "Charlie", "design", 1, true),
        (4, "Migration Guide v3", "Diana", "guide", 1, false),
        (5, "Performance Tuning", "Alice", "guide", 5, true),
        (6, "Security Whitepaper", "Eve", "whitepaper", 2, true),
        (7, "Release Notes 1.2", "Frank", "release", 1, true),
        (8, "Troubleshooting FAQ", "Grace", "guide", 4, true),
    ];

    for (id, title, author, doc_type, version, published) in &expected {
        let row = query_rows(
            &db,
            &format!(
                "SELECT title, author, doc_type, version, published FROM documents WHERE id = {}",
                id
            ),
        );
        assert_eq!(row.len(), 1, "Document id={} should exist", id);
        assert_eq!(row[0][0], *title, "Document id={} title mismatch", id);
        assert_eq!(row[0][1], *author, "Document id={} author mismatch", id);
        assert_eq!(row[0][2], *doc_type, "Document id={} doc_type mismatch", id);
        let ver: i64 = row[0][3].parse().unwrap();
        assert_eq!(ver, *version, "Document id={} version mismatch", id);
        let pub_val = row[0][4] == "true" || row[0][4] == "1";
        assert_eq!(pub_val, *published, "Document id={} published mismatch", id);
    }

    // Content should not be empty
    let content = query_str(&db, "SELECT content FROM documents WHERE id = 1");
    assert!(
        content.as_deref().unwrap().contains("Welcome"),
        "Document 1 content should contain 'Welcome'"
    );

    // updated_at nullable: some have values, some NULL
    let has_updated = query_i64(
        &db,
        "SELECT COUNT(*) FROM documents WHERE updated_at IS NOT NULL",
    );
    assert!(has_updated > 0);
    let no_updated = query_i64(
        &db,
        "SELECT COUNT(*) FROM documents WHERE updated_at IS NULL",
    );
    assert!(no_updated > 0);

    db.close().unwrap();
}

// ── Test: metrics table every row ───────────────────────────────────

#[test]
fn test_migration_v037_metrics_every_row() {
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = tmp.path().join("migrated_metrics");
    copy_v037db(&db_dir);

    let dsn = format!("file://{}", db_dir.display());
    let db = Database::open(&dsn).unwrap();

    assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 26);

    // Verify host distribution
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM metrics WHERE host = 'srv-01'"),
        9
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM metrics WHERE host = 'srv-02'"),
        9
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM metrics WHERE host = 'srv-03'"),
        7
    );
    assert_eq!(
        query_i64(&db, "SELECT COUNT(*) FROM metrics WHERE host = 'srv-04'"),
        1 // WAL insert
    );

    // Verify metric types: cpu_usage(10+1 WAL), memory_used(6), disk_io(3), network_rx(3), network_tx(3)
    assert_eq!(
        query_i64(
            &db,
            "SELECT COUNT(*) FROM metrics WHERE metric_name = 'cpu_usage'"
        ),
        11 // 10 original + 1 WAL (srv-04)
    );
    assert_eq!(
        query_i64(
            &db,
            "SELECT COUNT(*) FROM metrics WHERE metric_name = 'memory_used'"
        ),
        6
    );

    // Spot-check values
    let cpu = query_f64(&db, "SELECT value FROM metrics WHERE id = 1");
    assert!((cpu - 45.2).abs() < 0.01, "Metric 1 value = {}", cpu);

    let mem = query_f64(&db, "SELECT value FROM metrics WHERE id = 7");
    assert!((mem - 14336.0).abs() < 0.01, "Metric 7 value = {}", mem);

    // WAL-inserted metric
    let wal_metric = query_f64(&db, "SELECT value FROM metrics WHERE id = 26");
    assert!(
        (wal_metric - 33.3).abs() < 0.01,
        "WAL metric value = {}",
        wal_metric
    );

    // JSON tags
    let tag = query_str(
        &db,
        "SELECT JSON_EXTRACT(tags, '$.core') FROM metrics WHERE id = 1",
    );
    assert_eq!(tag.as_deref(), Some("0"));

    db.close().unwrap();
}
