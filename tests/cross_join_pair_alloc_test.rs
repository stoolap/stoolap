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

//! A hash join's probe row that pairs with many build rows is copied for
//! each pair by a count, not by its values, counted per run on its thread

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

/// The allocations fifty more pairs take, as counted on the commit that
/// shared the probe row: the joined rows' own
const FIFTY_PAIRS: usize = 101;

#[test]
fn a_probe_row_paired_with_many_build_rows_is_not_copied_per_pair() {
    let db = Database::open("memory://cross_join_pair_alloc").unwrap();
    for sql in [
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
        "INSERT INTO users SELECT g.value, 'user' || g.value FROM generate_series(1, 200) g",
        "INSERT INTO orders SELECT g.value, CASE g.value % 4 WHEN 0 THEN 'new' \
         WHEN 1 THEN 'paid' WHEN 2 THEN 'shipped' ELSE 'done' END \
         FROM generate_series(1, 400) g",
        "ANALYZE users",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let query = db
        .prepare(
            "SELECT u.name, o.status FROM users u \
             CROSS JOIN (SELECT DISTINCT status FROM orders) o LIMIT $1",
        )
        .unwrap();
    let run = |limit: i64| {
        let before = ALLOCATIONS.with(Cell::get);
        let rows = query.query((limit,)).unwrap().count();
        assert_eq!(rows, limit as usize);
        ALLOCATIONS.with(Cell::get) - before
    };
    run(100);
    for _ in 0..3 {
        let fifty = run(100) - run(50);
        assert!(
            fifty <= FIFTY_PAIRS,
            "{fifty} allocations for fifty more pairs, {FIFTY_PAIRS} expected"
        );
    }
}
