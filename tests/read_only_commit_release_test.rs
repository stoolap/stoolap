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

//! A read-only transaction that is committed must release the per-table
//! stores it opened for its reads, as a rolled back or dropped one does.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;

use stoolap::Database;

/// Allocations minus deallocations, so growth over a loop is retention
static LIVE: AtomicI64 = AtomicI64::new(0);

/// The counter is process-wide, so the tests in this file never overlap
static SERIAL: Mutex<()> = Mutex::new(());

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(1, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Live allocations gained over `reps` explicit transactions that read the
/// table and end with `commit` or, when false, rollback
fn retained_over(db: &Database, reps: usize, commit: bool) -> i64 {
    let run = || {
        let mut tx = db.begin().unwrap();
        let n = tx
            .query("SELECT v FROM t WHERE id = 1", ())
            .unwrap()
            .count();
        assert_eq!(n, 1);
        if commit {
            tx.commit().unwrap();
        } else {
            tx.rollback().unwrap();
        }
    };
    for _ in 0..200 {
        run();
    }
    let before = LIVE.load(Ordering::Relaxed);
    for _ in 0..reps {
        run();
    }
    LIVE.load(Ordering::Relaxed) - before
}

#[test]
fn a_committed_read_only_transaction_releases_its_table_stores() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let db = Database::open("memory://read_only_commit_release").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10)", ()).unwrap();
    let reps = 1000;
    let rolled_back = retained_over(&db, reps, false);
    let committed = retained_over(&db, reps, true);
    eprintln!("live allocations gained over {reps} read-only transactions: rollback {rolled_back}, commit {committed}");
    assert!(
        committed <= rolled_back + 8,
        "{reps} committed read-only transactions retained {committed} allocations, rolled back ones {rolled_back}"
    );
}
