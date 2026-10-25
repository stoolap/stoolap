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

//! An ORDER BY with a LIMIT whose keys a projection computes allocates for
//! the rows it keeps, not the rows its WHERE passes, counted in the
//! allocations the query makes on its thread

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

/// The rows of `sql`'s second run and the allocations it took
fn allocations(db: &Database, sql: &str) -> (usize, usize) {
    let rows = |db: &Database| db.query(sql, ()).unwrap().count();
    rows(db);
    let before = ALLOCATIONS.with(Cell::get);
    let count = rows(db);
    (count, ALLOCATIONS.with(Cell::get) - before)
}

#[cfg(not(feature = "test-filedb"))]
#[test]
fn an_order_by_limit_with_computed_keys_projects_the_rows_it_keeps() {
    let db = Database::open("memory://general_order_limit_alloc").unwrap();
    db.execute(
        "CREATE TABLE d (id INTEGER PRIMARY KEY, g INTEGER, s TEXT, f FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO d SELECT value, value % 40, 's' || value, value * 0.5 \
         FROM generate_series(1, 20000)",
        (),
    )
    .unwrap();
    db.execute("CREATE VIEW w AS SELECT id, g, s, f FROM d", ())
        .unwrap();
    let mut wrong = Vec::new();
    for sql in [
        "SELECT id, g FROM d WHERE f > 5000 ORDER BY g + 0, id LIMIT 10",
        "SELECT id, g + 1 AS h FROM d WHERE f > 5000 ORDER BY h, id LIMIT 10",
        "SELECT id, s FROM d WHERE f > 5000 ORDER BY g DESC, id LIMIT 10 OFFSET 5",
        "SELECT id FROM w WHERE f > 5000 ORDER BY g + 0, id LIMIT 10",
    ] {
        let (count, allocations) = allocations(&db, sql);
        if count != 10 || allocations >= 2000 {
            wrong.push(format!("{count} rows, {allocations} allocations: {sql}"));
        }
    }
    // A derived table's own read allocates per row; the sort adds no more
    let (_, plain) = allocations(
        &db,
        "SELECT id, g FROM (SELECT id, g, f FROM d) x WHERE f > 5000 ORDER BY id LIMIT 10",
    );
    let sql =
        "SELECT id, g FROM (SELECT id, g, f FROM d) x WHERE f > 5000 ORDER BY g + 0, id LIMIT 10";
    let (count, computed) = allocations(&db, sql);
    if count != 10 || computed >= plain + 2000 {
        wrong.push(format!(
            "{count} rows, {computed} allocations against {plain} ordered by a column: {sql}"
        ));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
