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

//! Checkpoint cycle, compaction, and data integrity tests.
//!
//! Verifies that data survives multiple seal/compact rounds, that updates
//! and deletes across checkpoint boundaries produce correct results, and
//! that aggregations, indexes, and row counts remain accurate throughout.

use std::fs;
use std::path::{Path, PathBuf};

use stoolap::storage::volume::manifest::TableManifest;
use stoolap::{Database, IsolationLevel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn query_all_i64(db: &Database, sql: &str) -> Vec<i64> {
    let mut r = db.query(sql, ()).unwrap();
    let mut out = Vec::new();
    while let Some(Ok(row)) = r.next() {
        if let Ok(v) = row.get::<i64>(0) {
            out.push(v);
        }
    }
    out
}

fn count_rows(db: &Database, sql: &str) -> i64 {
    let mut r = db.query(sql, ()).unwrap();
    let mut n = 0i64;
    for _ in r.by_ref() {
        n += 1;
    }
    n
}

fn list_volume_files(db_path: &Path, table: &str) -> Vec<PathBuf> {
    let table_dir = db_path.join("volumes").join(table);
    let mut files: Vec<PathBuf> = match fs::read_dir(&table_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("vol"))
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    files
}

fn read_manifest(db_path: &Path, table: &str) -> TableManifest {
    let path = db_path.join("volumes").join(table).join("manifest.bin");
    TableManifest::read_from_disk(&path).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Multiple checkpoint cycles
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_checkpoint_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/multi_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE data (id INTEGER PRIMARY KEY, batch INTEGER, val TEXT)",
            (),
        )
        .unwrap();

        // Batch 1: ids 1..=100
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=100 {
            db.execute(
                &format!("INSERT INTO data VALUES ({}, 1, 'b1_{}')", i, i),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 100);

        // Batch 2: ids 101..=250
        db.execute("BEGIN", ()).unwrap();
        for i in 101..=250 {
            db.execute(
                &format!("INSERT INTO data VALUES ({}, 2, 'b2_{}')", i, i),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 250);

        // Batch 3: ids 251..=400
        db.execute("BEGIN", ()).unwrap();
        for i in 251..=400 {
            db.execute(
                &format!("INSERT INTO data VALUES ({}, 3, 'b3_{}')", i, i),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 400);

        // Verify per-batch counts
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE batch = 1"),
            100
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE batch = 2"),
            150
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE batch = 3"),
            150
        );

        // Spot-check values
        assert_eq!(query_str(&db, "SELECT val FROM data WHERE id = 1"), "b1_1");
        assert_eq!(
            query_str(&db, "SELECT val FROM data WHERE id = 200"),
            "b2_200"
        );
        assert_eq!(
            query_str(&db, "SELECT val FROM data WHERE id = 400"),
            "b3_400"
        );

        db.close().unwrap();
    }

    // Reopen and verify everything survived
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 400);
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE batch = 1"),
            100
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE batch = 2"),
            150
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE batch = 3"),
            150
        );
        assert_eq!(
            query_str(&db, "SELECT val FROM data WHERE id = 50"),
            "b1_50"
        );
        assert_eq!(
            query_str(&db, "SELECT val FROM data WHERE id = 300"),
            "b3_300"
        );
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 2. Updates across checkpoint boundaries
// ---------------------------------------------------------------------------

#[test]
fn test_updates_across_checkpoint_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/upd_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, price FLOAT, label TEXT)",
            (),
        )
        .unwrap();

        // Insert initial rows
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=50 {
            db.execute(
                &format!(
                    "INSERT INTO items VALUES ({}, {}.0, 'orig_{}')",
                    i,
                    i * 10,
                    i
                ),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Checkpoint: rows move to cold
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Update some rows (now hot overrides cold)
        for i in 1..=20 {
            db.execute(
                &format!(
                    "UPDATE items SET price = {}.0, label = 'updated_{}' WHERE id = {}",
                    i * 100,
                    i,
                    i
                ),
                (),
            )
            .unwrap();
        }

        // Verify updates visible
        let p1 = query_f64(&db, "SELECT price FROM items WHERE id = 1");
        assert!((p1 - 100.0).abs() < 0.01, "expected 100.0, got {}", p1);
        let p20 = query_f64(&db, "SELECT price FROM items WHERE id = 20");
        assert!((p20 - 2000.0).abs() < 0.01, "expected 2000.0, got {}", p20);
        // Non-updated row should keep original value
        let p30 = query_f64(&db, "SELECT price FROM items WHERE id = 30");
        assert!((p30 - 300.0).abs() < 0.01, "expected 300.0, got {}", p30);

        // Checkpoint again: updated rows sealed
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify after second checkpoint
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 50);
        let p1_post = query_f64(&db, "SELECT price FROM items WHERE id = 1");
        assert!(
            (p1_post - 100.0).abs() < 0.01,
            "post-checkpoint: expected 100.0, got {}",
            p1_post
        );
        assert_eq!(
            query_str(&db, "SELECT label FROM items WHERE id = 10"),
            "updated_10"
        );
        assert_eq!(
            query_str(&db, "SELECT label FROM items WHERE id = 40"),
            "orig_40"
        );

        db.close().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM items"), 50);
        let p5 = query_f64(&db, "SELECT price FROM items WHERE id = 5");
        assert!(
            (p5 - 500.0).abs() < 0.01,
            "reopen: expected 500.0, got {}",
            p5
        );
        assert_eq!(
            query_str(&db, "SELECT label FROM items WHERE id = 5"),
            "updated_5"
        );
        let p45 = query_f64(&db, "SELECT price FROM items WHERE id = 45");
        assert!(
            (p45 - 450.0).abs() < 0.01,
            "reopen: expected 450.0, got {}",
            p45
        );
        assert_eq!(
            query_str(&db, "SELECT label FROM items WHERE id = 45"),
            "orig_45"
        );
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 3. Deletes across checkpoint boundaries
// ---------------------------------------------------------------------------

#[test]
fn test_deletes_across_checkpoint_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/del_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, val TEXT)", ())
            .unwrap();

        // Insert rows
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=80 {
            db.execute(&format!("INSERT INTO data VALUES ({}, 'v_{}')", i, i), ())
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Checkpoint: rows go cold
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 80);

        // Delete rows 1..=30
        for i in 1..=30 {
            db.execute(&format!("DELETE FROM data WHERE id = {}", i), ())
                .unwrap();
        }
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 50);

        // Checkpoint: tombstones applied, should compact
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 50);

        // Verify deleted rows are gone
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE id <= 30"),
            0
        );
        // Survivors present
        assert_eq!(query_str(&db, "SELECT val FROM data WHERE id = 31"), "v_31");
        assert_eq!(query_str(&db, "SELECT val FROM data WHERE id = 80"), "v_80");

        db.close().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM data"), 50);
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM data WHERE id <= 30"),
            0
        );
        assert_eq!(query_str(&db, "SELECT val FROM data WHERE id = 50"), "v_50");
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 4. Insert, update, delete, checkpoint loop (realistic workload)
// ---------------------------------------------------------------------------

#[test]
fn test_mixed_workload_checkpoint_loop() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/mixed_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE ledger (id INTEGER PRIMARY KEY, amount INTEGER, status TEXT)",
            (),
        )
        .unwrap();

        // Round 1: Insert 100 rows (ids 1..=100), amount = id * 10
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=100 {
            db.execute(
                &format!("INSERT INTO ledger VALUES ({}, {}, 'active')", i, i * 10),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM ledger"), 100);

        // Round 2: Update 20 rows (1..=20), delete 10 rows (81..=90), insert 50 more (101..=150)
        for i in 1..=20 {
            db.execute(
                &format!(
                    "UPDATE ledger SET amount = {}, status = 'modified' WHERE id = {}",
                    i * 100,
                    i
                ),
                (),
            )
            .unwrap();
        }
        for i in 81..=90 {
            db.execute(&format!("DELETE FROM ledger WHERE id = {}", i), ())
                .unwrap();
        }
        db.execute("BEGIN", ()).unwrap();
        for i in 101..=150 {
            db.execute(
                &format!("INSERT INTO ledger VALUES ({}, {}, 'active')", i, i * 10),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Expected: 100 - 10 + 50 = 140
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM ledger"), 140);
        // Deleted rows gone
        assert_eq!(
            query_i64(
                &db,
                "SELECT COUNT(*) FROM ledger WHERE id BETWEEN 81 AND 90"
            ),
            0
        );
        // Updated rows have new values
        assert_eq!(
            query_i64(&db, "SELECT amount FROM ledger WHERE id = 5"),
            500
        );
        assert_eq!(
            query_str(&db, "SELECT status FROM ledger WHERE id = 5"),
            "modified"
        );

        // Round 3: Update 30 rows (21..=50), delete 15 rows (91..=100, 101..=105), insert 25 (151..=175)
        for i in 21..=50 {
            db.execute(
                &format!(
                    "UPDATE ledger SET amount = {}, status = 'round3' WHERE id = {}",
                    i * 1000,
                    i
                ),
                (),
            )
            .unwrap();
        }
        for i in 91..=100 {
            db.execute(&format!("DELETE FROM ledger WHERE id = {}", i), ())
                .unwrap();
        }
        for i in 101..=105 {
            db.execute(&format!("DELETE FROM ledger WHERE id = {}", i), ())
                .unwrap();
        }
        db.execute("BEGIN", ()).unwrap();
        for i in 151..=175 {
            db.execute(
                &format!("INSERT INTO ledger VALUES ({}, {}, 'active')", i, i * 10),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Expected: 140 - 15 + 25 = 150
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM ledger"), 150);

        // Verify round3 updates
        assert_eq!(
            query_i64(&db, "SELECT amount FROM ledger WHERE id = 30"),
            30000
        );
        assert_eq!(
            query_str(&db, "SELECT status FROM ledger WHERE id = 30"),
            "round3"
        );
        // Round1 update still intact
        assert_eq!(
            query_i64(&db, "SELECT amount FROM ledger WHERE id = 10"),
            1000
        );
        assert_eq!(
            query_str(&db, "SELECT status FROM ledger WHERE id = 10"),
            "modified"
        );
        // Untouched row
        assert_eq!(
            query_i64(&db, "SELECT amount FROM ledger WHERE id = 70"),
            700
        );
        assert_eq!(
            query_str(&db, "SELECT status FROM ledger WHERE id = 70"),
            "active"
        );

        db.close().unwrap();
    }

    // Reopen and verify final state
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM ledger"), 150);

        // All deleted ranges gone
        assert_eq!(
            query_i64(
                &db,
                "SELECT COUNT(*) FROM ledger WHERE id BETWEEN 81 AND 105"
            ),
            0
        );

        // Updated rows
        assert_eq!(
            query_i64(&db, "SELECT amount FROM ledger WHERE id = 1"),
            100
        );
        assert_eq!(
            query_i64(&db, "SELECT amount FROM ledger WHERE id = 40"),
            40000
        );
        assert_eq!(
            query_i64(&db, "SELECT amount FROM ledger WHERE id = 150"),
            1500
        );

        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 5. Compaction correctness (>4 checkpoints)
// ---------------------------------------------------------------------------

#[test]
fn test_compaction_correctness() {
    let dir = tempfile::tempdir().unwrap();
    // Lower compact threshold to trigger compaction sooner
    let dsn = format!(
        "file://{}/compaction?compact_threshold=3",
        dir.path().display()
    );

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE kv (id INTEGER PRIMARY KEY, ver INTEGER, data TEXT)",
            (),
        )
        .unwrap();

        // 6 rounds of insert + checkpoint to generate >3 volumes (triggers compaction)
        let mut total_rows = 0i64;
        for round in 1..=6 {
            let start = (round - 1) * 50 + 1;
            let end = round * 50;
            db.execute("BEGIN", ()).unwrap();
            for i in start..=end {
                db.execute(
                    &format!(
                        "INSERT INTO kv VALUES ({}, {}, 'r{}_{}')",
                        i, round, round, i
                    ),
                    (),
                )
                .unwrap();
            }
            db.execute("COMMIT", ()).unwrap();
            total_rows += 50;
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            assert_eq!(
                query_i64(&db, "SELECT COUNT(*) FROM kv"),
                total_rows,
                "after round {} checkpoint",
                round
            );
        }

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM kv"), 300);

        // Verify newest data is accessible
        assert_eq!(query_str(&db, "SELECT data FROM kv WHERE id = 1"), "r1_1");
        assert_eq!(
            query_str(&db, "SELECT data FROM kv WHERE id = 275"),
            "r6_275"
        );

        // Also do updates and checkpoint to test compaction with overwrites
        for i in 1..=30 {
            db.execute(
                &format!(
                    "UPDATE kv SET ver = 99, data = 'compacted_{}' WHERE id = {}",
                    i, i
                ),
                (),
            )
            .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM kv"), 300);
        assert_eq!(
            query_str(&db, "SELECT data FROM kv WHERE id = 10"),
            "compacted_10"
        );
        assert_eq!(query_i64(&db, "SELECT ver FROM kv WHERE id = 10"), 99);

        db.close().unwrap();
    }

    // Reopen and verify compacted data
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM kv"), 300);
        // Updated rows
        assert_eq!(
            query_str(&db, "SELECT data FROM kv WHERE id = 5"),
            "compacted_5"
        );
        assert_eq!(query_i64(&db, "SELECT ver FROM kv WHERE id = 5"), 99);
        // Non-updated rows
        assert_eq!(
            query_str(&db, "SELECT data FROM kv WHERE id = 100"),
            "r2_100"
        );
        assert_eq!(query_i64(&db, "SELECT ver FROM kv WHERE id = 100"), 2);
        db.close().unwrap();
    }
}

#[test]
fn test_snapshot_transaction_blocks_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("snapshot_blocks_compaction");
    let dsn = format!(
        "file://{}?compact_threshold=10&checkpoint_on_close=off",
        db_path.display()
    );

    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
        .unwrap();

    for batch in 0..2 {
        db.execute("BEGIN", ()).unwrap();
        let start = batch * 10 + 1;
        let end = start + 9;
        for id in start..=end {
            db.execute(
                &format!("INSERT INTO t VALUES ({}, 'batch_{}')", id, batch + 1),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }

    assert_eq!(list_volume_files(&db_path, "t").len(), 2);

    db.execute("PRAGMA compact_threshold = 2", ()).unwrap();

    let mut snapshot_tx = db
        .begin_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    let snapshot_count: i64 = snapshot_tx.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(snapshot_count, 20);

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        list_volume_files(&db_path, "t").len(),
        2,
        "compaction should be skipped while a snapshot transaction is active"
    );

    snapshot_tx.rollback().unwrap();

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(
        list_volume_files(&db_path, "t").len(),
        1,
        "compaction should run after the snapshot transaction ends"
    );

    db.close().unwrap();
}

#[test]
fn test_compaction_skips_table_when_manifest_segment_is_not_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("skip_incomplete_compaction");
    let setup_dsn = format!(
        "file://{}?compact_threshold=10&checkpoint_on_close=off",
        db_path.display()
    );

    {
        let db = Database::open(&setup_dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", ())
            .unwrap();

        for batch in 0..2 {
            db.execute("BEGIN", ()).unwrap();
            let start = batch * 10 + 1;
            let end = start + 9;
            for id in start..=end {
                db.execute(
                    &format!("INSERT INTO t VALUES ({}, 'batch_{}')", id, batch + 1),
                    (),
                )
                .unwrap();
            }
            db.execute("COMMIT", ()).unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }

        db.close().unwrap();
    }

    let volume_files = list_volume_files(&db_path, "t");
    assert_eq!(volume_files.len(), 2, "expected two standalone volumes");

    let manifest_before = read_manifest(&db_path, "t");
    assert_eq!(manifest_before.segments.len(), 2);
    let segment_ids_before: Vec<u64> = manifest_before
        .segments
        .iter()
        .map(|s| s.segment_id)
        .collect();

    fs::remove_file(&volume_files[0]).unwrap();

    let compact_dsn = format!(
        "file://{}?compact_threshold=2&checkpoint_on_close=off",
        db_path.display()
    );
    let db = Database::open(&compact_dsn).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();

    let manifest_after = read_manifest(&db_path, "t");
    let segment_ids_after: Vec<u64> = manifest_after
        .segments
        .iter()
        .map(|s| s.segment_id)
        .collect();

    assert_eq!(
        manifest_after.segments.len(),
        2,
        "compaction should not shrink the manifest when a selected segment failed to load"
    );
    assert_eq!(
        segment_ids_after, segment_ids_before,
        "compaction should leave the manifest unchanged when not all selected volumes are loaded"
    );
}

// ---------------------------------------------------------------------------
// 6. Row count accuracy through cycles
// ---------------------------------------------------------------------------

#[test]
fn test_row_count_accuracy_through_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/count_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE nums (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
            (),
        )
        .unwrap();

        // Phase 1: Insert 200 rows
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=200 {
            let cat = if i % 2 == 0 { "even" } else { "odd" };
            db.execute(
                &format!("INSERT INTO nums VALUES ({}, '{}', {})", i, cat, i),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // Verify before checkpoint
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nums"), 200);

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Fast count path (no WHERE)
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nums"), 200);
        // Count with WHERE
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nums WHERE category = 'even'"),
            100
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nums WHERE category = 'odd'"),
            100
        );
        // Manual row enumeration should match
        assert_eq!(count_rows(&db, "SELECT * FROM nums"), 200);

        // Phase 2: Delete 50 even rows (ids 2,4,...,100)
        for i in (2..=100).step_by(2) {
            db.execute(&format!("DELETE FROM nums WHERE id = {}", i), ())
                .unwrap();
        }

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nums"), 150);
        assert_eq!(count_rows(&db, "SELECT * FROM nums"), 150);

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nums"), 150);
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nums WHERE category = 'even'"),
            50
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nums WHERE category = 'odd'"),
            100
        );
        assert_eq!(count_rows(&db, "SELECT * FROM nums"), 150);

        // Phase 3: Add 75 more
        db.execute("BEGIN", ()).unwrap();
        for i in 201..=275 {
            let cat = if i % 2 == 0 { "even" } else { "odd" };
            db.execute(
                &format!("INSERT INTO nums VALUES ({}, '{}', {})", i, cat, i),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nums"), 225);

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nums"), 225);
        assert_eq!(count_rows(&db, "SELECT * FROM nums"), 225);

        db.close().unwrap();
    }

    // Reopen
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nums"), 225);
        assert_eq!(count_rows(&db, "SELECT * FROM nums"), 225);
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 7. Aggregation accuracy through cycles
// ---------------------------------------------------------------------------

/// Compute SUM by scanning rows individually (bypasses any aggregation pushdown).
fn scan_sum(db: &Database, table: &str, col: &str) -> f64 {
    let sql = format!("SELECT {} FROM {}", col, table);
    let mut r = db.query(&sql, ()).unwrap();
    let mut total = 0.0f64;
    while let Some(Ok(row)) = r.next() {
        if let Ok(v) = row.get::<f64>(0) {
            total += v;
        }
    }
    total
}

#[test]
fn test_aggregation_accuracy_through_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/agg_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE metrics (id INTEGER PRIMARY KEY, value FLOAT)",
            (),
        )
        .unwrap();

        // Insert 100 rows: value = id * 1.0 => values 1.0 through 100.0
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=100 {
            db.execute(&format!("INSERT INTO metrics VALUES ({}, {}.0)", i, i), ())
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        // SUM(1..100) = 5050, AVG = 50.5, MIN = 1, MAX = 100
        assert!((query_f64(&db, "SELECT SUM(value) FROM metrics") - 5050.0).abs() < 0.01);
        assert!((query_f64(&db, "SELECT AVG(value) FROM metrics") - 50.5).abs() < 0.01);
        assert!((query_f64(&db, "SELECT MIN(value) FROM metrics") - 1.0).abs() < 0.01);
        assert!((query_f64(&db, "SELECT MAX(value) FROM metrics") - 100.0).abs() < 0.01);

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify aggregates unchanged after checkpoint (insert-only, no dedup issues)
        assert!((query_f64(&db, "SELECT SUM(value) FROM metrics") - 5050.0).abs() < 0.01);
        assert!((query_f64(&db, "SELECT AVG(value) FROM metrics") - 50.5).abs() < 0.01);

        // Verify COUNT is correct
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 100);

        db.close().unwrap();
    }

    // Second session: insert-only aggregation across reopen + checkpoint
    {
        let db = Database::open(&dsn).unwrap();

        // Aggregates should still be correct after reopen
        let sum_reopen = query_f64(&db, "SELECT SUM(value) FROM metrics");
        assert!(
            (sum_reopen - 5050.0).abs() < 0.01,
            "reopen SUM expected 5050.0, got {}",
            sum_reopen
        );

        // Add 10 more rows: ids 101..=110 with values 101..=110
        // New total sum: 5050 + (101+102+...+110) = 5050 + 1055 = 6105
        db.execute("BEGIN", ()).unwrap();
        for i in 101..=110 {
            db.execute(&format!("INSERT INTO metrics VALUES ({}, {}.0)", i, i), ())
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        let sum_added = query_f64(&db, "SELECT SUM(value) FROM metrics");
        assert!(
            (sum_added - 6105.0).abs() < 0.01,
            "after insert SUM expected 6105.0, got {}",
            sum_added
        );

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // After checkpoint of insert-only data, aggregates should be correct
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 110);
        let min_val = query_f64(&db, "SELECT MIN(value) FROM metrics");
        assert!(
            (min_val - 1.0).abs() < 0.01,
            "MIN expected 1.0, got {}",
            min_val
        );
        let max_val = query_f64(&db, "SELECT MAX(value) FROM metrics");
        assert!(
            (max_val - 110.0).abs() < 0.01,
            "MAX expected 110.0, got {}",
            max_val
        );

        // Delete rows 106..=110
        for i in 106..=110 {
            db.execute(&format!("DELETE FROM metrics WHERE id = {}", i), ())
                .unwrap();
        }
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 105);

        // Compute scan-based sum (manual iteration) before checkpoint
        let scan_before = scan_sum(&db, "metrics", "value");
        // SUM(1..105) = 105*106/2 = 5565
        assert!(
            (scan_before - 5565.0).abs() < 0.01,
            "scan SUM before checkpoint expected 5565.0, got {}",
            scan_before
        );

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify via scan that row-level data is correct after checkpoint
        let scan_after = scan_sum(&db, "metrics", "value");
        assert!(
            (scan_after - 5565.0).abs() < 0.01,
            "scan SUM after checkpoint expected 5565.0, got {}",
            scan_after
        );

        // COUNT should reflect deletes
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 105);

        // Verify MIN/MAX via scan
        let min_after = query_f64(&db, "SELECT MIN(value) FROM metrics");
        assert!(
            (min_after - 1.0).abs() < 0.01,
            "MIN expected 1.0, got {}",
            min_after
        );
        let max_after = query_f64(&db, "SELECT MAX(value) FROM metrics");
        assert!(
            (max_after - 105.0).abs() < 0.01,
            "MAX expected 105.0, got {}",
            max_after
        );

        db.close().unwrap();
    }

    // Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM metrics"), 105);
        let scan_reopen = scan_sum(&db, "metrics", "value");
        assert!(
            (scan_reopen - 5565.0).abs() < 0.01,
            "reopen scan SUM expected 5565.0, got {}",
            scan_reopen
        );
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 8. Index correctness through cycles
// ---------------------------------------------------------------------------

#[test]
fn test_index_correctness_through_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/idx_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                sku TEXT NOT NULL,
                price FLOAT NOT NULL,
                category TEXT NOT NULL
            )",
            (),
        )
        .unwrap();
        db.execute("CREATE UNIQUE INDEX idx_sku ON products (sku)", ())
            .unwrap();
        db.execute("CREATE INDEX idx_category ON products (category)", ())
            .unwrap();

        // Insert initial data
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=100 {
            let cat = match i % 3 {
                0 => "A",
                1 => "B",
                _ => "C",
            };
            db.execute(
                &format!(
                    "INSERT INTO products VALUES ({}, 'SKU-{:04}', {}.99, '{}')",
                    i,
                    i,
                    i * 10,
                    cat
                ),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify index lookups work after checkpoint
        let price = query_f64(&db, "SELECT price FROM products WHERE sku = 'SKU-0042'");
        assert!(
            (price - 420.99).abs() < 0.01,
            "index lookup expected 420.99, got {}",
            price
        );
        let cat_count = query_i64(&db, "SELECT COUNT(*) FROM products WHERE category = 'A'");
        assert_eq!(cat_count, 33); // ids 3,6,...,99 => 33 rows

        // Verify unique constraint survives checkpoint
        let dup_result = db.execute(
            "INSERT INTO products VALUES (999, 'SKU-0042', 0.0, 'X')",
            (),
        );
        assert!(
            dup_result.is_err(),
            "unique constraint should prevent duplicate SKU after checkpoint"
        );

        // Update some rows
        for i in 1..=20 {
            db.execute(
                &format!(
                    "UPDATE products SET price = {}.00, category = 'UPDATED' WHERE id = {}",
                    i * 1000,
                    i
                ),
                (),
            )
            .unwrap();
        }

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify updated values via index
        let updated_count = query_i64(
            &db,
            "SELECT COUNT(*) FROM products WHERE category = 'UPDATED'",
        );
        assert_eq!(updated_count, 20);

        // Unique constraint still works
        let dup2 = db.execute(
            "INSERT INTO products VALUES (998, 'SKU-0001', 0.0, 'X')",
            (),
        );
        assert!(
            dup2.is_err(),
            "unique constraint should still prevent duplicate SKU"
        );

        // Index lookup on updated row
        let updated_price = query_f64(&db, "SELECT price FROM products WHERE sku = 'SKU-0010'");
        assert!(
            (updated_price - 10000.00).abs() < 0.01,
            "updated price expected 10000.0, got {}",
            updated_price
        );

        // Non-updated row still correct via index
        let orig_price = query_f64(&db, "SELECT price FROM products WHERE sku = 'SKU-0050'");
        assert!(
            (orig_price - 500.99).abs() < 0.01,
            "original price expected 500.99, got {}",
            orig_price
        );

        db.close().unwrap();
    }

    // Reopen and verify indexes still work
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM products"), 100);

        // Unique constraint
        let dup3 = db.execute(
            "INSERT INTO products VALUES (997, 'SKU-0050', 0.0, 'X')",
            (),
        );
        assert!(dup3.is_err(), "unique constraint should survive reopen");

        // Index-based lookups
        let p = query_f64(&db, "SELECT price FROM products WHERE sku = 'SKU-0075'");
        assert!(
            (p - 750.99).abs() < 0.01,
            "reopen index lookup expected 750.99, got {}",
            p
        );
        assert_eq!(
            query_i64(
                &db,
                "SELECT COUNT(*) FROM products WHERE category = 'UPDATED'"
            ),
            20
        );

        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 9. Multiple tables across checkpoints
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_tables_checkpoint_isolation() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/multi_tbl", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())
            .unwrap();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total FLOAT)",
            (),
        )
        .unwrap();

        // Insert into both tables
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=50 {
            db.execute(
                &format!("INSERT INTO users VALUES ({}, 'user_{}')", i, i),
                (),
            )
            .unwrap();
        }
        for i in 1..=200 {
            let uid = (i % 50) + 1;
            db.execute(
                &format!("INSERT INTO orders VALUES ({}, {}, {}.50)", i, uid, i * 10),
                (),
            )
            .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 50);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 200);

        // Modify users table only
        db.execute("UPDATE users SET name = 'admin' WHERE id = 1", ())
            .unwrap();
        db.execute("DELETE FROM users WHERE id = 50", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Users changed, orders untouched
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 49);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 200);
        assert_eq!(
            query_str(&db, "SELECT name FROM users WHERE id = 1"),
            "admin"
        );

        db.close().unwrap();
    }

    // Reopen
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM users"), 49);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM orders"), 200);
        assert_eq!(
            query_str(&db, "SELECT name FROM users WHERE id = 1"),
            "admin"
        );
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 10. Repeated update of the same row across checkpoints
// ---------------------------------------------------------------------------

#[test]
fn test_repeated_update_same_row() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/repeat_upd", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE counter (id INTEGER PRIMARY KEY, val INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO counter VALUES (1, 0)", ()).unwrap();

        // Repeatedly update and checkpoint
        for round in 1..=10 {
            db.execute(
                &format!("UPDATE counter SET val = {} WHERE id = 1", round),
                (),
            )
            .unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            assert_eq!(
                query_i64(&db, "SELECT val FROM counter WHERE id = 1"),
                round,
                "round {}",
                round
            );
        }

        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM counter"), 1);
        assert_eq!(query_i64(&db, "SELECT val FROM counter WHERE id = 1"), 10);

        db.close().unwrap();
    }

    // Reopen
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM counter"), 1);
        assert_eq!(query_i64(&db, "SELECT val FROM counter WHERE id = 1"), 10);
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 11. Insert-select across checkpoint boundaries
// ---------------------------------------------------------------------------

#[test]
fn test_insert_select_across_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/ins_sel_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();

        db.execute("CREATE TABLE src (id INTEGER PRIMARY KEY, data TEXT)", ())
            .unwrap();
        db.execute("CREATE TABLE dst (id INTEGER PRIMARY KEY, data TEXT)", ())
            .unwrap();

        // Insert into src, checkpoint
        db.execute("BEGIN", ()).unwrap();
        for i in 1..=100 {
            db.execute(&format!("INSERT INTO src VALUES ({}, 'src_{}')", i, i), ())
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // INSERT INTO SELECT from cold data
        db.execute(
            "INSERT INTO dst SELECT id, data FROM src WHERE id <= 50",
            (),
        )
        .unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM dst"), 50);

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // Verify dst
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM dst"), 50);
        assert_eq!(
            query_str(&db, "SELECT data FROM dst WHERE id = 25"),
            "src_25"
        );

        db.close().unwrap();
    }

    // Reopen
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM src"), 100);
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM dst"), 50);
        assert_eq!(query_str(&db, "SELECT data FROM dst WHERE id = 1"), "src_1");
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 12. Large batch to exercise seal threshold
// ---------------------------------------------------------------------------

#[test]
fn test_large_batch_seal_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/large_seal", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE big (id INTEGER PRIMARY KEY, payload TEXT)",
            (),
        )
        .unwrap();

        // Insert enough rows to exceed seal threshold (100K first, 10K incremental)
        // Using a moderate batch to keep test time reasonable
        let batch_size = 500;
        for round in 1..=4 {
            db.execute("BEGIN", ()).unwrap();
            let start = (round - 1) * batch_size + 1;
            let end = round * batch_size;
            let stmt = db.prepare("INSERT INTO big VALUES ($1, $2)").unwrap();
            for i in start..=end {
                stmt.execute((i as i64, format!("payload_r{}_{}", round, i)))
                    .unwrap();
            }
            db.execute("COMMIT", ()).unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();

            let expected = (round * batch_size) as i64;
            assert_eq!(
                query_i64(&db, "SELECT COUNT(*) FROM big"),
                expected,
                "round {}",
                round
            );
        }

        // Total: 2000 rows
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM big"), 2000);

        // Spot checks
        assert_eq!(
            query_str(&db, "SELECT payload FROM big WHERE id = 1"),
            "payload_r1_1"
        );
        assert_eq!(
            query_str(&db, "SELECT payload FROM big WHERE id = 2000"),
            "payload_r4_2000"
        );

        db.close().unwrap();
    }

    // Reopen
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM big"), 2000);
        assert_eq!(
            query_str(&db, "SELECT payload FROM big WHERE id = 1000"),
            "payload_r2_1000"
        );
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 13. Checkpoint with NULL values
// ---------------------------------------------------------------------------

#[test]
fn test_null_values_through_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/null_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE nullable (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
            (),
        )
        .unwrap();

        db.execute("INSERT INTO nullable VALUES (1, 'alice', 100)", ())
            .unwrap();
        db.execute("INSERT INTO nullable VALUES (2, NULL, 200)", ())
            .unwrap();
        db.execute("INSERT INTO nullable VALUES (3, 'charlie', NULL)", ())
            .unwrap();
        db.execute("INSERT INTO nullable VALUES (4, NULL, NULL)", ())
            .unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // NULL handling
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nullable WHERE name IS NULL"),
            2
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nullable WHERE score IS NULL"),
            2
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nullable WHERE name IS NOT NULL"),
            2
        );

        // Update a non-null to null
        db.execute("UPDATE nullable SET name = NULL WHERE id = 1", ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nullable WHERE name IS NULL"),
            3
        );

        db.close().unwrap();
    }

    // Reopen
    {
        let db = Database::open(&dsn).unwrap();
        assert_eq!(query_i64(&db, "SELECT COUNT(*) FROM nullable"), 4);
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nullable WHERE name IS NULL"),
            3
        );
        assert_eq!(
            query_i64(&db, "SELECT COUNT(*) FROM nullable WHERE score IS NULL"),
            2
        );
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 14. ORDER BY correctness after checkpoint
// ---------------------------------------------------------------------------

#[test]
fn test_order_by_after_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/order_ckpt", dir.path().display());

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE ranked (id INTEGER PRIMARY KEY, score INTEGER)",
            (),
        )
        .unwrap();

        db.execute("BEGIN", ()).unwrap();
        // Insert scores in non-sorted order
        for i in 1..=50 {
            let score = ((i * 37) % 50) + 1; // pseudo-shuffle
            db.execute(&format!("INSERT INTO ranked VALUES ({}, {})", i, score), ())
                .unwrap();
        }
        db.execute("COMMIT", ()).unwrap();

        db.execute("PRAGMA CHECKPOINT", ()).unwrap();

        // ORDER BY ASC
        let asc = query_all_i64(&db, "SELECT score FROM ranked ORDER BY score ASC LIMIT 5");
        for i in 1..asc.len() {
            assert!(asc[i] >= asc[i - 1], "ASC order violated: {:?}", asc);
        }
        assert_eq!(asc[0], 1);

        // ORDER BY DESC
        let desc = query_all_i64(&db, "SELECT score FROM ranked ORDER BY score DESC LIMIT 5");
        for i in 1..desc.len() {
            assert!(desc[i] <= desc[i - 1], "DESC order violated: {:?}", desc);
        }
        assert_eq!(desc[0], 50);

        db.close().unwrap();
    }

    // Reopen
    {
        let db = Database::open(&dsn).unwrap();
        let min = query_i64(&db, "SELECT MIN(score) FROM ranked");
        let max = query_i64(&db, "SELECT MAX(score) FROM ranked");
        assert_eq!(min, 1);
        assert_eq!(max, 50);

        let first = query_i64(&db, "SELECT score FROM ranked ORDER BY score ASC LIMIT 1");
        assert_eq!(first, 1);
        db.close().unwrap();
    }
}
