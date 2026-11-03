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

//! A join side read from its source moves the rows it generates for the run
//! on as they are, copying none of them

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counters are const
    // thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.try_with(Cell::get).unwrap_or(false) {
            ALLOCATIONS.with(|a| a.set(a.get() + 1));
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: ThreadCounting = ThreadCounting;

/// The allocations of one run of `sql` with `rows` in place of `{n}`
fn allocations(db: &Database, sql: &str, rows: usize) -> f64 {
    let statement = db.prepare(&sql.replace("{n}", &rows.to_string())).unwrap();
    for _ in 0..3 {
        statement.query(()).unwrap().count();
    }
    let before = ALLOCATIONS.with(Cell::get);
    COUNTING.with(|c| c.set(true));
    for _ in 0..5 {
        statement.query(()).unwrap().count();
    }
    COUNTING.with(|c| c.set(false));
    (ALLOCATIONS.with(Cell::get) - before) as f64 / 5.0
}

#[test]
fn a_generated_side_costs_a_joined_row_no_copy_of_its_own() {
    let db = Database::open("memory://join_source_side_alloc").unwrap();
    let mut wrong = Vec::new();
    for sql in [
        "SELECT s.n, v.k FROM generate_series(1, {n}) AS s(n) CROSS JOIN (VALUES (1)) AS v(k)",
        "SELECT s.n, v.k FROM generate_series(1, {n}) AS s(n) CROSS JOIN (VALUES (1)) AS v(k) \
         WHERE s.n > 0",
    ] {
        let per_row = (allocations(&db, sql, 2000) - allocations(&db, sql, 1000)) / 1000.0;
        // The series' row and the joined row; a copy of each row adds one
        if per_row >= 2.5 {
            wrong.push(format!("{sql}: {per_row} allocations per row"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
