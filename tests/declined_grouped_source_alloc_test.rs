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

//! A grouping whose table declines to compute its groups, inside a
//! transaction that wrote rows of its own, builds nothing for the groups it
//! did not get: each call it adds allocates what it allocates in a grouping
//! that never asks the table

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

/// The allocations of one run of `sql`, prepared, inside a transaction that
/// inserted a row of its own into t
fn allocations(name: &str, sql: &str) -> usize {
    let db = Database::open(&format!("memory://declined_grouped_source_{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value % 10, value FROM generate_series(1, 200)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (1000, 3, 7)", ()).unwrap();
    assert_eq!(tx.query(sql, ()).unwrap().count(), 10, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    tx.query(sql, ()).unwrap().count();
    let used = ALLOCATIONS.with(Cell::get) - before;
    tx.rollback().unwrap();
    used
}

#[test]
fn a_declined_table_grouping_allocates_nothing_for_its_calls() {
    let declined = (
        allocations("declined_one", "SELECT k, SUM(a) FROM t GROUP BY k"),
        allocations(
            "declined_three",
            "SELECT k, SUM(a), MIN(a), MAX(a) FROM t GROUP BY k",
        ),
    );
    let never_asked = (
        allocations(
            "never_one",
            "SELECT k, SUM(a) FROM t WHERE id = id GROUP BY k",
        ),
        allocations(
            "never_three",
            "SELECT k, SUM(a), MIN(a), MAX(a) FROM t WHERE id = id GROUP BY k",
        ),
    );
    assert_eq!(
        declined.1 - declined.0,
        never_asked.1 - never_asked.0,
        "declined {declined:?}, never asked {never_asked:?}"
    );
}
