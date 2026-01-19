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

//! Dirty read reproduction test

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use stoolap::api::Database;

#[test]
fn test_dirty_read_reproduction() {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Setup: Create table with initial value
    db.execute(
        "CREATE TABLE dirty_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO dirty_test VALUES (1, 100)", ())
        .expect("Failed to insert initial data");

    // Verify initial value
    let initial: i64 = db
        .query_one("SELECT value FROM dirty_test WHERE id = 1", ())
        .expect("Failed to query initial value");
    assert_eq!(initial, 100, "Initial value should be 100");

    let dirty_read_count = Arc::new(AtomicI32::new(0));
    let total_iterations = 10; // Reduced for faster testing

    for iteration in 0..total_iterations {
        // Clone BEFORE spawning threads to keep references alive
        let db_writer = db.clone();
        let db_reader = db.clone();
        let dirty_count = Arc::clone(&dirty_read_count);

        let writer_started = Arc::new(std::sync::Barrier::new(2));
        let writer_started_clone = Arc::clone(&writer_started);

        // Thread A: Writer that updates then rolls back
        let writer_handle = thread::spawn(move || {
            // Begin transaction
            db_writer.execute("BEGIN", ()).expect("BEGIN failed");

            // Update to 999 (uncommitted)
            db_writer
                .execute("UPDATE dirty_test SET value = 999 WHERE id = 1", ())
                .expect("UPDATE failed");

            // Signal that update is done
            writer_started_clone.wait();

            // Wait a bit to give reader time to read
            thread::sleep(Duration::from_micros(500));

            // Rollback - 999 was never committed!
            db_writer.execute("ROLLBACK", ()).expect("ROLLBACK failed");
        });

        // Thread B: Reader that tries to read during the uncommitted update
        let reader_handle = thread::spawn(move || {
            // Wait for writer to complete the UPDATE
            writer_started.wait();

            // Small delay to ensure we're reading during the uncommitted state
            thread::sleep(Duration::from_micros(100));

            // Read the value - should see 100 (committed), NOT 999 (uncommitted)
            let value: Result<i64, _> =
                db_reader.query_one("SELECT value FROM dirty_test WHERE id = 1", ());

            if let Ok(v) = value {
                if v == 999 {
                    // DIRTY READ DETECTED!
                    dirty_count.fetch_add(1, Ordering::SeqCst);
                    eprintln!(
                        "DIRTY READ in iteration {}: Read uncommitted value 999!",
                        iteration
                    );
                }
            }
        });

        writer_handle.join().expect("Writer thread panicked");
        reader_handle.join().expect("Reader thread panicked");
    }

    let dirty_reads = dirty_read_count.load(Ordering::SeqCst);
    println!(
        "\nDirty read test completed: {}/{} dirty reads detected",
        dirty_reads, total_iterations
    );

    // Verify final value is still 100 (all updates were rolled back)
    let final_value: i64 = db
        .query_one("SELECT value FROM dirty_test WHERE id = 1", ())
        .expect("Failed to query final value");
    assert_eq!(
        final_value, 100,
        "Final value should be 100 after rollbacks"
    );

    // This assertion will fail if dirty reads are detected
    assert_eq!(
        dirty_reads, 0,
        "Dirty reads detected! {} out of {} iterations had dirty reads",
        dirty_reads, total_iterations
    );
}

#[test]
fn test_dirty_read_simple() {
    // Simpler version with explicit transaction handles
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE simple_dirty (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO simple_dirty VALUES (1, 100)", ())
        .expect("Failed to insert");

    // Start a transaction and update (but don't commit)
    let mut tx = db.begin().expect("Failed to begin transaction");
    tx.execute("UPDATE simple_dirty SET value = 999 WHERE id = 1", ())
        .expect("Failed to update");

    // Read from outside the transaction - should see 100, not 999
    let value: i64 = db
        .query_one("SELECT value FROM simple_dirty WHERE id = 1", ())
        .expect("Failed to query");

    println!("Value read outside transaction: {}", value);

    if value == 999 {
        eprintln!("DIRTY READ: Saw uncommitted value 999!");
    }

    // Rollback the transaction
    tx.rollback().expect("Failed to rollback");

    // Verify value is still 100
    let final_value: i64 = db
        .query_one("SELECT value FROM simple_dirty WHERE id = 1", ())
        .expect("Failed to query final");

    assert_eq!(final_value, 100, "Value should be 100 after rollback");
    assert_eq!(
        value, 100,
        "DIRTY READ DETECTED: Read uncommitted value {} instead of 100",
        value
    );
}

#[test]
fn test_shared_executor_issue() {
    // This test demonstrates that db.clone() shares the same executor
    // and thus the same active transaction state
    let db = Database::open_in_memory().expect("Failed to create database");

    db.execute(
        "CREATE TABLE shared_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO shared_test VALUES (1, 100)", ())
        .expect("Failed to insert");

    let db2 = db.clone();

    // Start transaction on db
    db.execute("BEGIN", ()).expect("BEGIN failed");
    db.execute("UPDATE shared_test SET value = 999 WHERE id = 1", ())
        .expect("UPDATE failed");

    // Read from db2 (cloned) - this SHOULD see 100 (committed value)
    // but if executor is shared, it will see 999 (uncommitted)
    let value: i64 = db2
        .query_one("SELECT value FROM shared_test WHERE id = 1", ())
        .expect("Failed to query");

    println!(
        "db2 read value: {} (expected 100 if isolated, 999 if shared)",
        value
    );

    db.execute("ROLLBACK", ()).expect("ROLLBACK failed");

    // After fix: db.clone() creates independent executor with its own transaction state
    // so db2 should see the committed value (100), not the uncommitted value (999)
    if value == 100 {
        println!("CORRECT: db2 sees committed value 100, not uncommitted 999");
    } else {
        eprintln!(
            "BUG: db2 read uncommitted value {} from db's transaction",
            value
        );
    }

    // After the fix, db2 should see the committed value (100)
    assert_eq!(
        value, 100,
        "db2 should see committed value 100, not uncommitted value from db's transaction"
    );
}
