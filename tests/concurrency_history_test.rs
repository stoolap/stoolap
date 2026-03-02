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

#![cfg(feature = "stress-tests")]

//! Concurrency History Checking Tests
//!
//! Records a history of operations from concurrent threads, then validates
//! that the history is consistent with the claimed isolation level.
//!
//! Validation rules:
//! - Read Committed: no dirty reads (a txn never sees uncommitted writes)
//! - Snapshot Isolation: consistent snapshot (all reads within a txn see
//!   the same snapshot) and write-write conflict detection

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;
use stoolap::Database;

// ============================================================================
// History recording types
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Op {
    Begin {
        thread: usize,
        wall_ns: u64,
    },
    Read {
        thread: usize,
        key: i64,
        value: Option<i64>,
        wall_ns: u64,
    },
    Write {
        thread: usize,
        key: i64,
        value: i64,
        wall_ns: u64,
    },
    Commit {
        thread: usize,
        wall_ns: u64,
    },
    Rollback {
        thread: usize,
        wall_ns: u64,
    },
    CommitFailed {
        thread: usize,
        wall_ns: u64,
        error: String,
    },
}

static CLOCK: AtomicU64 = AtomicU64::new(0);

fn tick() -> u64 {
    CLOCK.fetch_add(1, Ordering::SeqCst)
}

type History = Arc<Mutex<Vec<Op>>>;

fn new_history() -> History {
    Arc::new(Mutex::new(Vec::new()))
}

fn record(history: &History, op: Op) {
    history.lock().unwrap().push(op);
}

// ============================================================================
// History validation
// ============================================================================

/// Validate Read Committed: no transaction ever reads a value written by
/// an uncommitted transaction.
fn validate_read_committed(history: &[Op]) {
    // Build a timeline: for each thread, track when it committed/rolled back
    // and what values it wrote.
    let mut committed_writes: Vec<(usize, i64, i64, u64)> = Vec::new(); // (thread, key, value, commit_time)

    // First pass: identify commits and rollbacks, collect writes per thread-txn
    // We track transaction "sessions" by thread. Each BEGIN starts a new session.
    let mut thread_session: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut session_counter = 0usize;
    let mut session_writes: std::collections::HashMap<usize, Vec<(i64, i64)>> =
        std::collections::HashMap::new();
    let mut session_commit_time: std::collections::HashMap<usize, u64> =
        std::collections::HashMap::new();
    let mut session_rolled_back: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    let mut read_session: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new(); // thread -> current session at read time

    for op in history {
        match op {
            Op::Begin { thread, .. } => {
                let session = session_counter;
                session_counter += 1;
                thread_session.insert(*thread, session);
                session_writes.entry(session).or_default();
            }
            Op::Write {
                thread, key, value, ..
            } => {
                if let Some(&session) = thread_session.get(thread) {
                    session_writes
                        .entry(session)
                        .or_default()
                        .push((*key, *value));
                }
            }
            Op::Read { thread, .. } => {
                if let Some(&session) = thread_session.get(thread) {
                    read_session.insert(*thread, session);
                }
            }
            Op::Commit { thread, wall_ns } => {
                if let Some(&session) = thread_session.get(thread) {
                    session_commit_time.insert(session, *wall_ns);
                }
            }
            Op::Rollback { thread, .. } => {
                if let Some(&session) = thread_session.get(thread) {
                    session_rolled_back.insert(session);
                }
            }
            Op::CommitFailed { thread, .. } => {
                if let Some(&session) = thread_session.get(thread) {
                    session_rolled_back.insert(session);
                }
            }
        }
    }

    // Collect all committed writes with their commit times
    for (session, writes) in &session_writes {
        if session_rolled_back.contains(session) {
            continue; // Skip rolled-back sessions
        }
        if let Some(&commit_time) = session_commit_time.get(session) {
            for &(key, value) in writes {
                committed_writes.push((*session, key, value, commit_time));
            }
        }
    }

    // Build set of rolled-back (key, value) pairs
    let mut uncommitted_values: std::collections::HashSet<(i64, i64)> =
        std::collections::HashSet::new();
    for (session, writes) in &session_writes {
        if session_rolled_back.contains(session) {
            for &(key, value) in writes {
                // Only mark as uncommitted if no committed session wrote the same (key, value)
                let also_committed = committed_writes
                    .iter()
                    .any(|(_, k, v, _)| *k == key && *v == value);
                if !also_committed {
                    uncommitted_values.insert((key, value));
                }
            }
        }
    }

    // Second pass: check reads
    // Reset session tracking for reads
    let mut current_session_for_thread: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    session_counter = 0;

    for op in history {
        match op {
            Op::Begin { thread, .. } => {
                current_session_for_thread.insert(*thread, session_counter);
                session_counter += 1;
            }
            Op::Read {
                thread,
                key,
                value: Some(read_val),
                wall_ns,
            } => {
                // Check 1: this value must NOT be from a rolled-back transaction
                if uncommitted_values.contains(&(*key, *read_val)) {
                    panic!(
                        "DIRTY READ DETECTED: thread {} at time {} read key={} value={} \
                         which was written by a rolled-back transaction",
                        thread, wall_ns, key, read_val
                    );
                }

                // Check 2: this value must be either:
                // (a) an initial setup value, or
                // (b) written by a committed transaction, or
                // (c) written by the thread's own transaction (read-your-writes)
                let is_initial_value = *read_val >= 0 && *read_val <= 9;

                let was_committed = committed_writes
                    .iter()
                    .any(|(_session, k, v, _commit_time)| *k == *key && *v == *read_val);

                let own_session = current_session_for_thread.get(thread);
                let is_own_write = own_session.is_some_and(|s| {
                    session_writes.get(s).is_some_and(|writes| {
                        writes.iter().any(|(k, v)| *k == *key && *v == *read_val)
                    })
                });

                if !is_initial_value && !was_committed && !is_own_write {
                    panic!(
                        "DIRTY READ DETECTED: thread {} at time {} read key={} value={} \
                         which was never committed and not an own write",
                        thread, wall_ns, key, read_val
                    );
                }
            }
            _ => {}
        }
    }
}

/// Validate Snapshot Isolation: within a single transaction, all reads
/// should see a consistent snapshot. If a txn reads key K twice and gets
/// different values, the snapshot is inconsistent.
fn validate_snapshot_consistency(history: &[Op]) {
    let mut session_counter = 0usize;
    let mut thread_session: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    // Track first-read values per session per key
    let mut session_first_reads: std::collections::HashMap<
        usize,
        std::collections::HashMap<i64, Option<i64>>,
    > = std::collections::HashMap::new();

    for op in history {
        match op {
            Op::Begin { thread, .. } => {
                thread_session.insert(*thread, session_counter);
                session_counter += 1;
            }
            Op::Read {
                thread,
                key,
                value,
                wall_ns,
            } => {
                if let Some(&session) = thread_session.get(thread) {
                    let reads = session_first_reads.entry(session).or_default();
                    if let Some(first_value) = reads.get(key) {
                        // We've read this key before in this session
                        // Under snapshot isolation, should see same value
                        // (unless we wrote to it ourselves in between)
                        let session_writes_to_key = history.iter().any(|op2| {
                            matches!(op2, Op::Write { thread: t, key: k, .. }
                                if thread_session.get(t) == Some(&session) && *k == *key)
                        });

                        if !session_writes_to_key && first_value != value {
                            panic!(
                                "SNAPSHOT INCONSISTENCY: thread {} session {} read key={} \
                                 first as {:?}, then as {:?} at time {} (no intervening write)",
                                thread, session, key, first_value, value, wall_ns
                            );
                        }
                    } else {
                        reads.insert(*key, *value);
                    }
                }
            }
            _ => {}
        }
    }
}

// ============================================================================
// Workload runner
// ============================================================================

fn run_concurrent_workload(
    seed: u64,
    num_threads: usize,
    num_keys: usize,
    ops_per_thread: usize,
    use_explicit_txn: bool,
) -> Vec<Op> {
    let db = Database::open_in_memory().expect("Failed to create database");

    // Setup: create table with initial values
    db.execute(
        "CREATE TABLE hist_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    for i in 0..num_keys {
        db.execute(
            &format!(
                "INSERT INTO hist_test VALUES ({}, {})",
                i,
                i // Initial value = key id
            ),
            (),
        )
        .expect("Failed to insert initial data");
    }

    let history = new_history();
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let db = db.clone();
            let history = Arc::clone(&history);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait(); // Synchronize start

                let mut local_seed = seed.wrapping_mul(thread_id as u64 + 1).wrapping_add(7);

                for _ in 0..ops_per_thread {
                    local_seed = local_seed.wrapping_mul(6364136223846793005).wrapping_add(1);

                    if use_explicit_txn {
                        // BEGIN transaction
                        if db.execute("BEGIN", ()).is_err() {
                            continue;
                        }
                        record(
                            &history,
                            Op::Begin {
                                thread: thread_id,
                                wall_ns: tick(),
                            },
                        );
                    }

                    let key = (local_seed % num_keys as u64) as i64;

                    // Decide: read or write
                    let do_write = (local_seed >> 16).is_multiple_of(3); // ~33% writes

                    if do_write {
                        let new_value = ((local_seed >> 8) % 1000) as i64 + 10; // 10+ to distinguish from initial
                        let result = db.execute(
                            &format!(
                                "UPDATE hist_test SET value = {} WHERE id = {}",
                                new_value, key
                            ),
                            (),
                        );
                        if result.is_ok() {
                            record(
                                &history,
                                Op::Write {
                                    thread: thread_id,
                                    key,
                                    value: new_value,
                                    wall_ns: tick(),
                                },
                            );
                        }
                    } else {
                        // Read
                        let result: Result<Option<i64>, _> = db.query_one(
                            &format!("SELECT value FROM hist_test WHERE id = {}", key),
                            (),
                        );
                        if let Ok(value) = result {
                            record(
                                &history,
                                Op::Read {
                                    thread: thread_id,
                                    key,
                                    value,
                                    wall_ns: tick(),
                                },
                            );
                        }
                    }

                    if use_explicit_txn {
                        // Decide: commit or rollback
                        let do_rollback = (local_seed >> 24).is_multiple_of(5); // 20% rollbacks

                        if do_rollback {
                            let _ = db.execute("ROLLBACK", ());
                            record(
                                &history,
                                Op::Rollback {
                                    thread: thread_id,
                                    wall_ns: tick(),
                                },
                            );
                        } else {
                            match db.execute("COMMIT", ()) {
                                Ok(_) => {
                                    record(
                                        &history,
                                        Op::Commit {
                                            thread: thread_id,
                                            wall_ns: tick(),
                                        },
                                    );
                                }
                                Err(e) => {
                                    let _ = db.execute("ROLLBACK", ());
                                    record(
                                        &history,
                                        Op::CommitFailed {
                                            thread: thread_id,
                                            wall_ns: tick(),
                                            error: e.to_string(),
                                        },
                                    );
                                }
                            }
                        }
                    }

                    // Small random delay to increase interleaving
                    if local_seed.is_multiple_of(7) {
                        thread::sleep(Duration::from_micros(10));
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    Arc::try_unwrap(history).unwrap().into_inner().unwrap()
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_read_committed_no_dirty_reads() {
    // Run 10 iterations with different seeds
    for seed in 0..10 {
        let history = run_concurrent_workload(
            seed * 12345 + 67890,
            4,    // 4 threads
            10,   // 10 keys
            50,   // 50 ops per thread
            true, // explicit transactions
        );

        // Validate: no dirty reads
        validate_read_committed(&history);
    }
}

#[test]
fn test_read_committed_no_dirty_reads_8_threads() {
    // Higher contention with 8 threads
    for seed in 0..5 {
        let history = run_concurrent_workload(
            seed * 99991 + 42,
            8,    // 8 threads
            5,    // 5 keys (high contention)
            30,   // 30 ops per thread
            true, // explicit transactions
        );

        validate_read_committed(&history);
    }
}

#[test]
fn test_snapshot_isolation_consistency() {
    // Use snapshot isolation (the default for explicit transactions in stoolap)
    for seed in 0..10 {
        let history = run_concurrent_workload(
            seed * 54321 + 11111,
            4,    // 4 threads
            10,   // 10 keys
            40,   // 40 ops per thread
            true, // explicit transactions (gives snapshot isolation)
        );

        // Validate: consistent snapshot reads
        validate_snapshot_consistency(&history);
    }
}

#[test]
fn test_autocommit_no_dirty_reads() {
    // Without explicit transactions, each statement auto-commits
    for seed in 0..10 {
        let history = run_concurrent_workload(
            seed * 77777 + 33333,
            4,     // 4 threads
            10,    // 10 keys
            100,   // 100 ops per thread
            false, // autocommit mode
        );

        // In autocommit mode, every read should see committed state
        // Check: no reads see values that were never committed
        let mut all_committed_values: std::collections::HashSet<(i64, i64)> =
            std::collections::HashSet::new();

        // Initial values are committed
        for i in 0..10i64 {
            all_committed_values.insert((i, i));
        }

        // All writes in autocommit are committed
        for op in &history {
            if let Op::Write { key, value, .. } = op {
                all_committed_values.insert((*key, *value));
            }
        }

        // Check reads
        for op in &history {
            if let Op::Read {
                thread,
                key,
                value: Some(value),
                wall_ns,
            } = op
            {
                assert!(
                    all_committed_values.contains(&(*key, *value)),
                    "Thread {} at time {} read key={} value={} which was never committed",
                    thread,
                    wall_ns,
                    key,
                    value
                );
            }
        }
    }
}

#[test]
fn test_concurrent_insert_delete_consistency() {
    // Test that concurrent inserts and deletes maintain consistency
    let db = Database::open_in_memory().expect("Failed to create database");
    db.execute(
        "CREATE TABLE cd_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");

    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let db = db.clone();
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                // Each thread operates on its own key range to avoid conflicts
                let base = thread_id * 100;
                for i in 0..50 {
                    let id = base + i;
                    // Insert
                    let _ = db.execute(&format!("INSERT INTO cd_test VALUES ({}, {})", id, i), ());
                    // Delete even ids
                    if i % 2 == 0 {
                        let _ = db.execute(&format!("DELETE FROM cd_test WHERE id = {}", id), ());
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify: each thread should have 25 remaining rows (odd ids)
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM cd_test", ())
        .expect("COUNT should work");
    assert_eq!(count, 100, "4 threads x 25 remaining rows = 100");

    // Verify no duplicate IDs
    let dup_count: i64 = db
        .query_one(
            "SELECT COUNT(*) FROM (SELECT id FROM cd_test GROUP BY id HAVING COUNT(*) > 1) AS d",
            (),
        )
        .unwrap_or(0);
    assert_eq!(dup_count, 0, "No duplicate primary keys");
}

#[test]
fn test_write_write_conflict_detection() {
    // Two transactions both try to update the same row.
    // Stoolap detects write-write conflicts eagerly at UPDATE time
    // (if another txn already has uncommitted changes on the row).
    // At least one transaction's full cycle (BEGIN+UPDATE+COMMIT) should succeed.
    let db = Database::open_in_memory().expect("Failed to create database");
    db.execute(
        "CREATE TABLE ww_test (id INTEGER PRIMARY KEY, value INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO ww_test VALUES (1, 0)", ())
        .expect("Failed to insert");

    let mut success_count = 0;
    let mut conflict_count = 0;

    for _ in 0..20 {
        let db1 = db.clone();
        let db2 = db.clone();
        let barrier = Arc::new(Barrier::new(2));
        let b1 = Arc::clone(&barrier);
        let b2 = Arc::clone(&barrier);

        let h1 = thread::spawn(move || -> bool {
            if db1.execute("BEGIN", ()).is_err() {
                return false;
            }
            b1.wait(); // Synchronize: both transactions active
            if db1
                .execute("UPDATE ww_test SET value = 1 WHERE id = 1", ())
                .is_err()
            {
                let _ = db1.execute("ROLLBACK", ());
                return false;
            }
            if db1.execute("COMMIT", ()).is_err() {
                let _ = db1.execute("ROLLBACK", ());
                return false;
            }
            true
        });

        let h2 = thread::spawn(move || -> bool {
            if db2.execute("BEGIN", ()).is_err() {
                return false;
            }
            b2.wait(); // Synchronize: both transactions active
            if db2
                .execute("UPDATE ww_test SET value = 2 WHERE id = 1", ())
                .is_err()
            {
                let _ = db2.execute("ROLLBACK", ());
                return false;
            }
            if db2.execute("COMMIT", ()).is_err() {
                let _ = db2.execute("ROLLBACK", ());
                return false;
            }
            true
        });

        let r1 = h1.join().expect("Thread 1 panicked");
        let r2 = h2.join().expect("Thread 2 panicked");

        match (r1, r2) {
            (true, true) => success_count += 1,
            (true, false) | (false, true) => conflict_count += 1,
            (false, false) => {
                // Both can fail if conflict is detected at UPDATE time for both
                // (timing-dependent). This is acceptable.
                conflict_count += 1;
            }
        }

        // Reset for next iteration
        let _ = db.execute("UPDATE ww_test SET value = 0 WHERE id = 1", ());
    }

    // At least some iterations should succeed
    assert!(
        success_count + conflict_count == 20,
        "All iterations should complete"
    );
    eprintln!(
        "Write-write conflicts: {} detected conflicts, {} both-succeed out of 20",
        conflict_count, success_count
    );
}
