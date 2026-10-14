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

//! A commit's cold recheck allocates nothing while it holds the
//! transaction's store, nor while it decides a round under the seal fence

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(feature = "test-failpoints", not(feature = "jemalloc")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::test_failpoints::{
    cold_rechecks, in_recheck_fence_section, in_recheck_store_lock, volume_loads, FailpointGuard,
};
use stoolap::Database;

struct Counting;

thread_local! {
    static UNDER_STORE_LOCK: Cell<usize> = const { Cell::new(0) };
    static UNDER_FENCE: Cell<usize> = const { Cell::new(0) };
}

impl Counting {
    fn count() {
        if in_recheck_store_lock() {
            let _ = UNDER_STORE_LOCK.try_with(|n| n.set(n.get() + 1));
        }
        if in_recheck_fence_section() {
            let _ = UNDER_FENCE.try_with(|n| n.set(n.get() + 1));
        }
    }
}

unsafe impl GlobalAlloc for Counting {
    // SAFETY: forwards to the system allocator; the flags and the counters
    // are const thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::count();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::count();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn counts() -> (usize, usize) {
    (
        UNDER_STORE_LOCK.with(Cell::get),
        UNDER_FENCE.with(Cell::get),
    )
}

#[test]
fn a_recheck_allocates_nothing_holding_the_store_or_deciding_under_the_fence() {
    let _guard = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 0)", ()).unwrap();
    let mut txn = db.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (2, 7, 0), (4, 8, 0)", ())
        .unwrap();
    // Rows 1 and 3 sealed together, so ids 2 and 4 fall in the volume's range
    db.execute("INSERT INTO t VALUES (3, 30, 0)", ()).unwrap();
    db.engine().seal_for_test().unwrap();
    db.engine().cold_volumes_for_test("t");
    let (rechecks, loads, before) = (cold_rechecks(), volume_loads().0, counts());
    txn.commit().unwrap();
    let after = counts();
    assert!(cold_rechecks() > rechecks, "the commit did not recheck");
    assert!(volume_loads().0 > loads, "the recheck loaded no volume");
    assert_eq!(
        after.0 - before.0,
        0,
        "allocations while the recheck held the transaction store"
    );
    assert_eq!(
        after.1 - before.1,
        0,
        "allocations while a recheck round was decided under the seal fence"
    );
}
