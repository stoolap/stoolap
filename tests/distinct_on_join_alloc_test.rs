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

//! An index join whose DISTINCT ON keys are SELECT items keeps its rows as
//! narrow as the join without DISTINCT ON, never the whole joined row, counted
//! in the allocations the query makes on its thread

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

const ROWS: usize = 2000;

fn allocations(db: &Database, sql: &str) -> (usize, usize) {
    assert!(db.query(sql, ()).unwrap().count() > 0, "{sql}");
    let before = ALLOCATIONS.with(Cell::get);
    let rows = db.query(sql, ()).unwrap().count();
    (ALLOCATIONS.with(Cell::get) - before, rows)
}

#[test]
fn distinct_on_over_selected_keys_keeps_the_index_join_narrow() {
    let db = Database::open("memory://distinct_on_join_alloc").unwrap();
    db.execute("CREATE TABLE v (id INTEGER PRIMARY KEY, a INTEGER)", ())
        .unwrap();
    db.execute(
        &format!("INSERT INTO v SELECT g.value, g.value % 50 FROM generate_series(1, {ROWS}) g"),
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE y (id INTEGER PRIMARY KEY, a INTEGER, pad TEXT, pad2 TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO y SELECT g.value, g.value % 40, REPEAT('x', 200), REPEAT('y', 200) \
             FROM generate_series(1, {ROWS}) g"
        ),
        (),
    )
    .unwrap();
    let (plain, joined) = allocations(&db, "SELECT y.a, v.id FROM v JOIN y ON y.id = v.id");
    assert_eq!(joined, ROWS);
    for sql in [
        "SELECT DISTINCT ON (y.a) y.a, v.id FROM v JOIN y ON y.id = v.id",
        "SELECT DISTINCT ON (1) y.a, v.id FROM v JOIN y ON y.id = v.id",
        "SELECT DISTINCT ON (k) y.a AS k, v.id FROM v JOIN y ON y.id = v.id",
    ] {
        let (distinct, rows) = allocations(&db, sql);
        assert_eq!(rows, 40, "{sql}");
        assert!(
            distinct < plain + ROWS / 2,
            "{sql}: {distinct} allocations, the join without DISTINCT ON {plain}"
        );
    }
}
