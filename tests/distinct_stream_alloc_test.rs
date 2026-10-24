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
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counters are const
    // thread-local Cells, which never allocate
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        BYTES.with(|b| b.set(b.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        BYTES.with(|b| b.set(b.get() + new_size));
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
    let (count, allocations, _) = allocated(db, sql);
    (count, allocations)
}

/// The rows of `sql`'s second run, and the allocations and bytes it took
fn allocated(db: &Database, sql: &str) -> (usize, usize, usize) {
    let rows = |db: &Database| db.query(sql, ()).unwrap().count();
    rows(db);
    let (allocations, bytes) = (ALLOCATIONS.with(Cell::get), BYTES.with(Cell::get));
    let count = rows(db);
    (
        count,
        ALLOCATIONS.with(Cell::get) - allocations,
        BYTES.with(Cell::get) - bytes,
    )
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

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_distinct_of_whole_rows_holds_only_the_rows_it_keeps() {
    let db = Database::open("memory://distinct_stream_alloc_whole_rows").unwrap();
    db.execute("CREATE TABLE w (g INTEGER)", ()).unwrap();
    db.execute(
        "INSERT INTO w SELECT value % 4 FROM generate_series(1, 100000)",
        (),
    )
    .unwrap();
    let (count, allocations, bytes) = allocated(
        &db,
        "SELECT DISTINCT g FROM w WHERE g >= (SELECT MIN(g) FROM w)",
    );
    assert_eq!(count, 4);
    assert!(
        bytes < 1_000_000,
        "{bytes} bytes, {allocations} allocations for 4 rows kept of 100000"
    );
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_streamed_distinct_dedups_its_rows_once() {
    let db = table("distinct_stream_alloc_once");
    let (count, allocations) = allocations(
        &db,
        "SELECT DISTINCT id FROM d WHERE id > (SELECT MIN(id) FROM d) - 1",
    );
    assert_eq!(count, 20000);
    assert!(
        allocations < 70_000,
        "{allocations} allocations for 20000 distinct rows"
    );
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_first_branch_whose_set_operation_dedups_holds_the_rows_it_keeps() {
    let db = table("distinct_stream_alloc_set_branch");
    for (sql, rows) in [
        (
            "SELECT g FROM d WHERE id > (SELECT MIN(id) FROM d) - 1 UNION SELECT g FROM d WHERE id < 3",
            4,
        ),
        (
            "SELECT g FROM d WHERE id > (SELECT MIN(id) FROM d) - 1 INTERSECT SELECT g FROM d WHERE id < 3",
            2,
        ),
        (
            "SELECT g FROM d WHERE id > (SELECT MIN(id) FROM d) - 1 EXCEPT SELECT g FROM d WHERE id < 3",
            2,
        ),
    ] {
        let (count, allocations) = allocations(&db, sql);
        assert_eq!(count, rows, "{sql}");
        assert!(
            allocations < 2000,
            "{allocations} allocations for {rows} rows kept of 20000: {sql}"
        );
    }
}
