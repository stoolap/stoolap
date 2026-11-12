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

//! A window argument read once and computed, a count or a frame offset,
//! costs a run only its one compile over no columns beside the same call
//! reading a parameter

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

/// The allocations of compiling `$1 + 1` over no columns
const COMPILE: usize = 4;

/// The allocations one run of `sql`, prepared, takes once it is warm
fn allocations(db: &Database, sql: &str) -> usize {
    let stmt = db.prepare(sql).unwrap();
    let run = || stmt.query((2,)).unwrap().count();
    run();
    assert_eq!(run(), 3, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_computed_window_argument_costs_its_compile_alone() {
    let db = Database::open("memory://window_argument_alloc").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    let mut wrong = Vec::new();
    for (computed, read) in [
        (
            "SELECT NTILE($1 + 1) OVER (ORDER BY id) FROM t",
            "SELECT NTILE($1) OVER (ORDER BY id) FROM t",
        ),
        (
            "SELECT SUM(a) OVER (ORDER BY id ROWS BETWEEN $1 + 1 PRECEDING AND CURRENT ROW) FROM t",
            "SELECT SUM(a) OVER (ORDER BY id ROWS BETWEEN $1 PRECEDING AND CURRENT ROW) FROM t",
        ),
    ] {
        let (computed_cost, read_cost) = (allocations(&db, computed), allocations(&db, read));
        if computed_cost > read_cost + COMPILE {
            wrong.push(format!(
                "{computed}: {computed_cost}, {read_cost} reading $1"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The allocations one warm run of `sql`, prepared, takes over 2000 rows
/// with `$1` a TEXT vector
fn vector_run(db: &Database, sql: &str) -> usize {
    let stmt = db.prepare(sql).unwrap();
    let run = || stmt.query(("[1.0, 2.0]",)).unwrap().count();
    assert_eq!(run(), 2_000, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn a_constant_nth_value_n_is_read_once_per_partition() {
    let db = Database::open("memory://window_argument_alloc_nth").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t SELECT value FROM generate_series(1, 2000)",
        (),
    )
    .unwrap();
    let computed = "SELECT NTH_VALUE(id, CAST(VEC_DISTANCE_L2($1, $1) AS INTEGER) + 1) \
                    OVER (ORDER BY id) FROM t";
    let written = "SELECT NTH_VALUE(id, 1) OVER (ORDER BY id), $1 IS NULL FROM t";
    let (computed_cost, written_cost) = (vector_run(&db, computed), vector_run(&db, written));
    assert!(
        computed_cost < written_cost + 200,
        "{computed}: {computed_cost}, {written_cost} with n written"
    );
}
