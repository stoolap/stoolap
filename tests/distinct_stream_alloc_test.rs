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

//! A prepared DISTINCT holds the rows it keeps, not the rows its WHERE
//! passes, counted in the allocations the query makes on its thread

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

fn table(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO d SELECT value, value % 4 FROM generate_series(1, 20000)",
        (),
    )
    .unwrap();
    db
}

fn allocations(db: &Database, sql: &str) -> (usize, usize) {
    let rows = |db: &Database| db.query(sql, ()).unwrap().count();
    rows(db);
    let before = ALLOCATIONS.with(Cell::get);
    let count = rows(db);
    (count, ALLOCATIONS.with(Cell::get) - before)
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_distinct_allocates_for_the_rows_it_keeps() {
    let db = table("distinct_stream_alloc_keeps");
    let (count, allocations) = allocations(
        &db,
        "SELECT DISTINCT g FROM d WHERE id > (SELECT MIN(id) FROM d) - 1",
    );
    assert_eq!(count, 4);
    assert!(
        allocations < 2000,
        "{allocations} allocations for 4 rows kept of 20000"
    );
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_distinct_limit_without_order_by_stops_at_its_rows() {
    let db = table("distinct_stream_alloc_limit");
    let (count, allocations) = allocations(
        &db,
        "SELECT DISTINCT id FROM d WHERE id > (SELECT MIN(id) FROM d) - 1 LIMIT 2",
    );
    assert_eq!(count, 2);
    assert!(
        allocations < 2000,
        "{allocations} allocations for 2 rows of 20000"
    );
}
