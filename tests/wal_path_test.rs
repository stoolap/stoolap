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
fn test_wal_path_discovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    eprintln!("DB path: {:?}", db_path);

    // Open database
    let db = Database::open(&dsn).unwrap();

    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO test VALUES (1)", ()).unwrap();

    // List the entire temp directory recursively
    eprintln!("\nAll files in temp directory:");
    fn list_recursive(path: &std::path::Path, indent: usize) {
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let metadata = entry.metadata().unwrap();
                let prefix = " ".repeat(indent);
                if metadata.is_dir() {
                    eprintln!("{}{:?}/", prefix, entry.file_name());
                    list_recursive(&entry.path(), indent + 2);
                } else {
                    eprintln!(
                        "{}{:?} ({} bytes)",
                        prefix,
                        entry.file_name(),
                        metadata.len()
                    );
                }
            }
        }
    }
    list_recursive(dir.path(), 0);

    db.close().unwrap();

    eprintln!("\nAfter close:");
    list_recursive(dir.path(), 0);

    // Reopen and verify data
    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
    assert_eq!(count, 1);
    eprintln!("\n✅ Data persisted correctly (count={})", count);
}
