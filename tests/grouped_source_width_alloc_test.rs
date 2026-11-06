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

//! A prepared grouped statement's run over a source of seventeen columns,
//! no name of them repeated, allocates at most once more than over sixteen

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

const RUNS: usize = 8;

/// The allocations of `RUNS` warm runs of `sql` over a table of an id and
/// `width` INTEGER columns c0, c1, ..., three rows of them
fn runs(case: usize, width: usize, sql: &str) -> usize {
    let db = Database::open(&format!("memory://grouped_source_width_{case}_{width}")).unwrap();
    let columns: Vec<String> = (0..width).map(|i| format!("c{i} INTEGER")).collect();
    db.execute(
        &format!(
            "CREATE TABLE w (id INTEGER PRIMARY KEY, {})",
            columns.join(", ")
        ),
        (),
    )
    .unwrap();
    let values: Vec<String> = (0..width).map(|i| i.to_string()).collect();
    for id in 1..=3 {
        db.execute(
            &format!("INSERT INTO w VALUES ({id}, {})", values.join(", ")),
            (),
        )
        .unwrap();
    }
    let stmt = db.prepare(sql).unwrap();
    for _ in 0..3 {
        stmt.query((0,)).unwrap().for_each(drop);
    }
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        stmt.query((0,)).unwrap().for_each(drop);
    }
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_grouped_run_over_a_wide_source_allocates_no_more_for_its_names() {
    for (case, sql) in [
        "SELECT c0, COUNT(*) FROM w WHERE id > $1 GROUP BY c0",
        "SELECT MAX(c1), COUNT(*) FROM w WHERE id > $1",
    ]
    .into_iter()
    .enumerate()
    {
        let (narrow, wide) = (runs(case, 15, sql), runs(case, 16, sql));
        // A run over seventeen columns pays one allocation more than over
        // sixteen whatever reads the names; checking them pays none
        assert!(
            wide <= narrow + RUNS,
            "{sql}: {narrow} allocations over 16 columns, {wide} over 17"
        );
    }
}
