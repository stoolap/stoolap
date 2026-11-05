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
fn test_wal_file_creation() {
    println!("Testing WAL creation...");
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let wal_path = db_path.with_extension("db.wal");
    let dsn = format!("file://{}", db_path.display());

    eprintln!("DB path: {:?}", db_path);
    eprintln!("Expected WAL path: {:?}", wal_path);

    // Phase 1: Create and insert
    {
        let db = Database::open(&dsn).unwrap();

        // Check directory contents after open
        eprintln!("Directory contents after open:");
        for entry in fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            eprintln!(
                "  {:?} - {} bytes",
                entry.file_name(),
                entry.metadata().unwrap().len()
            );
        }

        db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)", ())
            .unwrap();

        // Check after CREATE
        eprintln!("Directory contents after CREATE:");
        for entry in fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            eprintln!(
                "  {:?} - {} bytes",
                entry.file_name(),
                entry.metadata().unwrap().len()
            );
        }

        db.execute("INSERT INTO test (id, value) VALUES (1, 'hello')", ())
            .unwrap();

        // Check after INSERT
        eprintln!("Directory contents after INSERT:");
        for entry in fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            eprintln!(
                "  {:?} - {} bytes",
                entry.file_name(),
                entry.metadata().unwrap().len()
            );
        }

        db.close().unwrap();
    }

    // Check after close
    eprintln!("Directory contents after close:");
    for entry in fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        eprintln!(
            "  {:?} - {} bytes",
            entry.file_name(),
            entry.metadata().unwrap().len()
        );
    }

    // Reopen
    eprintln!("Reopening database...");
    let db = Database::open(&dsn).unwrap();

    let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
    eprintln!("Count after reopen: {}", count);

    assert_eq!(count, 1, "Data should persist after reopen");
}
