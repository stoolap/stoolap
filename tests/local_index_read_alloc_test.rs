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

//! An index read over a transaction's own rows costs the same allocations
//! for many candidates as for few

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

const RUNS: usize = 10;

/// The allocations per read of `sql`
fn per_read(db: &Database, sql: &str) -> usize {
    let statement = db.prepare(sql).unwrap();
    let read = || {
        for row in statement.query(()).unwrap() {
            row.unwrap();
        }
    };
    read();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..RUNS {
        read();
    }
    (ALLOCATIONS.with(Cell::get) - before) / RUNS
}

#[test]
fn an_index_read_of_many_own_rows_costs_what_one_of_few_costs() {
    let db = Database::open("memory://local_index_read_alloc").unwrap();
    for table in ["few", "many"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(&format!("CREATE INDEX {table}_k ON {table} (k)"), ())
            .unwrap();
    }
    db.execute("BEGIN", ()).unwrap();
    for (table, rows) in [("few", 2_000), ("many", 50_000)] {
        db.execute(
            &format!("INSERT INTO {table} SELECT value, 1, value FROM generate_series(1, {rows})"),
            (),
        )
        .unwrap();
    }
    let few = per_read(&db, "SELECT id FROM few WHERE k = 1 AND v < 0");
    let many = per_read(&db, "SELECT id FROM many WHERE k = 1 AND v < 0");
    db.execute("ROLLBACK", ()).unwrap();
    assert!(
        many <= few,
        "allocations per read of 2,000 and of 50,000 own candidates: {few}, {many}"
    );
}
