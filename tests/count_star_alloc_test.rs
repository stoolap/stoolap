// Copyright 2026 Stoolap Contributors
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

//! COUNT(*) that cannot take the stores' counters counts sealed rows by
//! their ids, building no row for them

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

const ROWS: i64 = 100_000;

#[test]
fn count_star_under_a_snapshot_builds_no_row_for_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, name TEXT)",
        (),
    )
    .unwrap();
    // One volume, read on this thread
    let values: Vec<String> = (1..=ROWS)
        .map(|id| format!("({id}, {}, 'n{id}')", id % 7))
        .collect();
    db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("DELETE FROM t WHERE id % 10 = 0", ()).unwrap();

    let mut tx = db
        .begin_with_isolation(stoolap::IsolationLevel::SnapshotIsolation)
        .unwrap();
    let count = |tx: &mut stoolap::api::Transaction| -> (i64, usize) {
        let before = ALLOCATIONS.with(Cell::get);
        let n: i64 = tx.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        (n, ALLOCATIONS.with(Cell::get) - before)
    };
    count(&mut tx);
    let (n, allocations) = count(&mut tx);
    assert_eq!(n, ROWS - ROWS / 10);
    assert!(
        allocations < 1_000,
        "COUNT(*) over {ROWS} sealed rows took {allocations} allocations"
    );
}
