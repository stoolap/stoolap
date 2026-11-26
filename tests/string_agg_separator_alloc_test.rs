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

//! STRING_AGG with ORDER BY turns each row's separator into text once: on
//! keys and values that tie, the sort compares separators already made
//! text, so what a call allocates grows with its rows, not its comparisons

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(all(not(feature = "jemalloc"), not(feature = "test-filedb")))]

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
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

/// The allocations of one run of `sql`, prepared, over `rows` rows that tie
/// on k and v, each with an INTEGER separator out of order
fn allocations(rows: i64, sql: &str) -> usize {
    let db = Database::open(&format!("memory://string_agg_separator_alloc_{rows}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v TEXT, sep INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT g.value, 1, 'v', (g.value * 7919) % 1000 \
             FROM generate_series(1, {rows}) g"
        ),
        (),
    )
    .unwrap();
    let query = db.prepare(sql).unwrap();
    let run = || query.query(()).unwrap().count();
    assert_eq!(run(), 1, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn tied_rows_allocate_by_their_count_not_their_comparisons() {
    for sql in [
        "SELECT STRING_AGG(v, sep ORDER BY k) FROM t",
        "SELECT GROUP_CONCAT(v, sep ORDER BY k) FROM t",
    ] {
        let [a, b, c] = [200, 400, 800].map(|rows| allocations(rows, sql));
        // Doubling the rows doubles what they add; a buffer's growth adds a few
        assert!(
            c - b <= 2 * (b - a) + 16,
            "{sql}: {a}, {b} and {c} allocations over 200, 400 and 800 rows"
        );
    }
}
