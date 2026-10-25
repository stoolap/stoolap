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

//! A set operation that removes duplicates copies a row only when its value
//! is new, counted in the allocations the query makes on its thread

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
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
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

/// The rows of `sql`'s second run and the allocations it took
fn allocations(db: &Database, sql: &str) -> (usize, usize) {
    let (count, allocations, _) = measured(db, sql);
    (count, allocations)
}

/// The rows of `sql`'s second run, its allocations and the bytes they asked
fn measured(db: &Database, sql: &str) -> (usize, usize, usize) {
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
fn a_deduplicating_set_operation_copies_the_rows_it_keeps() {
    let db = Database::open("memory://set_operation_alloc").unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO d SELECT value, value % 4 FROM generate_series(1, 20000)",
        (),
    )
    .unwrap();
    for (sql, rows) in [
        ("SELECT g FROM d UNION SELECT g FROM d WHERE id < 3", 4),
        ("SELECT g FROM d WHERE id < 3 UNION SELECT g FROM d", 4),
        ("SELECT g FROM d INTERSECT SELECT g FROM d WHERE id > 2", 4),
        ("SELECT g FROM d EXCEPT SELECT g FROM d WHERE id < 3", 2),
    ] {
        let (count, allocations) = allocations(&db, sql);
        assert_eq!(count, rows, "{sql}");
        assert!(
            allocations < 2000,
            "{allocations} allocations for {rows} values of 20000 rows: {sql}"
        );
    }
    let (_, grouped) = allocations(&db, "SELECT id FROM d GROUP BY id");
    let sql = "SELECT g FROM d WHERE id < 3 UNION SELECT id FROM d GROUP BY id";
    let (count, union) = allocations(&db, sql);
    assert_eq!(count, 20000, "{sql}");
    assert!(
        union < grouped + 2000,
        "{union} allocations against the branch's own {grouped}: {sql}"
    );
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn a_set_operation_keeping_no_row_asks_no_room_for_them() {
    let db = Database::open("memory://set_operation_room").unwrap();
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    // More rows than a pooled row buffer holds, so each buffer is asked anew
    db.execute(
        "INSERT INTO d SELECT value FROM generate_series(1, 70000)",
        (),
    )
    .unwrap();
    let (_, _, union) = measured(&db, "SELECT id FROM d UNION SELECT id FROM d WHERE id < 0");
    let sql = "SELECT id FROM d INTERSECT SELECT id FROM d WHERE id < 0";
    let (count, _, intersect) = measured(&db, sql);
    assert_eq!(count, 0, "{sql}");
    assert!(
        intersect < union + 64 * 1024,
        "{intersect} bytes against the UNION's {union}: {sql}"
    );
}
