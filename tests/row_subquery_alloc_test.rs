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

//! A row compared with an uncorrelated subquery's one row costs no
//! allocation per outer row, counted on the query's thread

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

const RUNS: usize = 20;

/// The allocations per run of `test` over `rows` outer rows
fn per_run(name: &str, test: &str, rows: usize) -> usize {
    let db = Database::open(&format!("memory://row_subquery_alloc_{name}_{rows}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        &format!("INSERT INTO t SELECT value, value, value FROM generate_series(1, {rows})"),
        (),
    )
    .unwrap();
    let statement = db
        .prepare(&format!("SELECT id FROM t WHERE {test}"))
        .unwrap();
    let run = || {
        for row in statement.query(()).unwrap() {
            row.unwrap();
        }
    };
    run();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        run();
    }
    (ALLOCATIONS.with(Cell::get) - before) / RUNS
}

#[test]
fn a_row_compared_with_no_subquery_row_allocates_nothing_per_outer_row() {
    for (name, test) in [
        (
            "distinct",
            "(a, b) IS NOT DISTINCT FROM (SELECT x, y FROM u)",
        ),
        ("ordered", "(a, b) < (SELECT x, y FROM u)"),
    ] {
        let (few, many) = (per_run(name, test, 100), per_run(name, test, 200));
        assert!(
            many.saturating_sub(few) < 50,
            "{test}: {few} allocations a run over 100 rows, {many} over 200"
        );
    }
}
