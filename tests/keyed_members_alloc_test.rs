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

//! A correlated IN block's members grouped by key cost a fixed number of
//! allocations per key, counted on the query's thread

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

#[test]
fn a_keyed_member_set_costs_four_allocations_per_key_not_five() {
    let mut costs = Vec::new();
    for keys in [100, 200] {
        let db = Database::open(&format!("memory://keyed_members_alloc_{keys}")).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
            (),
        )
        .unwrap();
        db.execute(
            &format!("INSERT INTO t SELECT value, value, value FROM generate_series(1, {keys})"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO u SELECT value, value % {keys}, value FROM generate_series(1, {})",
                keys * 3
            ),
            (),
        )
        .unwrap();
        let statement = db
            .prepare("SELECT id FROM t WHERE x IN (SELECT v FROM u WHERE u.k = t.k)")
            .unwrap();
        // A transaction keeps no answer between runs: each one groups again
        db.execute("BEGIN", ()).unwrap();
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
        costs.push((keys, (ALLOCATIONS.with(Cell::get) - before) / RUNS));
        db.execute("COMMIT", ()).unwrap();
    }
    assert!(
        costs[1].1 - costs[0].1 < 450,
        "(keys, allocations per run): {costs:?}"
    );
}
