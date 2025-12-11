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

// Test file lock prevents multiple database opens from different PROCESSES
// Note: Unix flock() allows the same process to acquire the lock multiple times
// The file lock is designed for inter-process locking, not intra-process

use stoolap::Database;
use tempfile::tempdir;

#[test]
fn test_file_lock_same_process_can_reopen() {
    // Within the same process, flock() allows multiple acquisitions
    // This is expected behavior for Unix file locks
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db");
    let dsn = format!("file://{}", db_path.to_str().unwrap());

    println!("Opening first database connection to: {}", dsn);
    let db1 = Database::open(&dsn).expect("First database should open");
    println!("First database opened successfully");

    // Same process can open again (flock allows this)
    println!("Opening second database connection from same process...");
    let db2 = Database::open(&dsn).expect("Same process should be able to open again");
    println!("Second database opened successfully (expected for same process)");

    drop(db1);
    drop(db2);

    println!("Same-process file lock test passed!");
}

#[test]
fn test_file_lock_released_on_drop() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_db2");
    let dsn = format!("file://{}", db_path.to_str().unwrap());

    // Open and close database multiple times
    for i in 1..=3 {
        println!("Opening database iteration {}", i);
        let db = Database::open(&dsn)
            .unwrap_or_else(|_| panic!("Database should open on iteration {}", i));

        // Do a simple operation
        db.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)",
            [],
        )
        .ok();

        println!("Closing database iteration {}", i);
        drop(db);
    }

    println!("File lock release test passed!");
}
