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

//! An index join decides whether its inner filter reads the key alone
//! without allocating, counted per run of a prepared query on its thread

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

/// A run's allocations as counted on the commit that made the check allocate nothing
const RUN_ALLOCATIONS: usize = 112;

#[test]
fn a_key_only_inner_filter_is_checked_without_allocating() {
    let db = Database::open("memory://join_key_filter_alloc").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 1), (2, 9), (3, 3)", ())
        .unwrap();
    let query = db
        .prepare("SELECT t.id, o.id FROM t JOIN o ON t.id = o.id WHERE t.a > 0 AND o.id IN (1, 3)")
        .unwrap();
    let run = || {
        let before = ALLOCATIONS.with(Cell::get);
        let rows = query.query(()).unwrap().count();
        (rows, ALLOCATIONS.with(Cell::get) - before)
    };
    run();
    for _ in 0..4 {
        let (rows, allocations) = run();
        assert_eq!(rows, 2);
        assert!(
            allocations <= RUN_ALLOCATIONS,
            "{allocations} allocations per run, {RUN_ALLOCATIONS} expected"
        );
    }
}
