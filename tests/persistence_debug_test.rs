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

use std::fs;
use stoolap::Database;
use tempfile::tempdir;

#[test]
fn test_persistence_debug() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    eprintln!("=== PHASE 1: Create and Insert ===");

    // Phase 1: Create table and insert data
    {
        let db = Database::open(&dsn).unwrap();
        eprintln!("Database opened");

        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        eprintln!("Table created");

        db.execute("INSERT INTO test (id, value) VALUES (1, 100)", ())
            .unwrap();
        eprintln!("Row 1 inserted");

        db.execute("INSERT INTO test (id, value) VALUES (2, 200)", ())
            .unwrap();
        eprintln!("Row 2 inserted");

        // Verify in phase 1
        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        eprintln!("Phase 1 count: {}", count);
        assert_eq!(count, 2);

        // List WAL files
        let wal_dir = db_path.join("wal");
        if wal_dir.exists() {
            eprintln!("WAL files:");
            for entry in fs::read_dir(&wal_dir).unwrap() {
                let entry = entry.unwrap();
                let content = fs::read(entry.path()).unwrap();
                eprintln!("  {:?} ({} bytes)", entry.file_name(), content.len());
                // Print first 200 bytes as hex
                let hex: Vec<String> = content
                    .iter()
                    .take(200)
                    .map(|b| format!("{:02x}", b))
                    .collect();
                eprintln!("  Content (first 200 bytes): {}", hex.join(" "));
            }
        } else {
            eprintln!("WAL directory doesn't exist!");
        }

        db.close().unwrap();
        eprintln!("Database closed");
    }

    // Check WAL after close
    eprintln!("\n=== After Phase 1 close ===");
    let wal_dir = db_path.join("wal");
    if wal_dir.exists() {
        eprintln!("WAL files after close:");
        for entry in fs::read_dir(&wal_dir).unwrap() {
            let entry = entry.unwrap();
            let content = fs::read(entry.path()).unwrap();
            eprintln!("  {:?} ({} bytes)", entry.file_name(), content.len());
        }
    }

    eprintln!("\n=== PHASE 2: Reopen and verify ===");

    // Phase 2: Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();
        eprintln!("Database reopened");

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        eprintln!("Phase 2 count: {}", count);

        if count == 2 {
            eprintln!("✅ SUCCESS: Data persisted correctly!");

            // Verify actual rows
            let rows = db
                .query("SELECT id, value FROM test ORDER BY id", ())
                .unwrap();
            for row in rows {
                let row = row.unwrap();
                let id: i64 = row.get(0).unwrap();
                let value: i64 = row.get(1).unwrap();
                eprintln!("  Row: id={}, value={}", id, value);
            }
        } else {
            eprintln!("❌ FAILURE: Expected 2 rows, got {}", count);
        }

        assert_eq!(count, 2, "Data should persist after reopen");
    }
}
